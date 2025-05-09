#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PCT_Pytorch.interpolation import Interpolation
from PCT_Pytorch.sampling import knn,weighted_random_point_sample,process_point_cloud_mix,weighted_random_point_sample,cal_weight
from pointnet2_ops import pointnet2_utils
from PCT_Pytorch.util import index_points


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    # print("pairwise_distance:",pairwise_distance.shape)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return -1*pairwise_distance,idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        _,idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        self.leaky_relu = bool(args.leaky_relu)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        ###add 
        self.sample_type = args.sample_type
        self.use_upsample = args.use_upsample

        if self.leaky_relu:
            act_mod = nn.LeakyReLU
            act_mod_args = {'negative_slope': 0.2}
        else:
            act_mod = nn.ReLU
            act_mod_args = {}

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   act_mod(**act_mod_args))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   act_mod(**act_mod_args))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   act_mod(**act_mod_args))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   act_mod(**act_mod_args))
        ###输入通道512，输出1024，卷积核的长度1，conv1d的参数[1024，512，1]（及1024（输出通道）个[512,1]的卷积核）
        ###输入是[B,512,1024]，每个batch在[512,1024]上用[512，1]卷积，得到[1,1024]，共有1024个卷积核所以做1024次，得到[1024,1024]
        ###Conv1d模型的卷积核大小是[输入通道数，卷积核的长]
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   act_mod(**act_mod_args))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    ###add normal
    def forward(self, x,normal):
        # print("x:",x.shape)
        # print("normal:",normal.shape)
        #add for use_upsample
        # print("self.use_upsample:",self.use_upsample)
        npoint=768
        xyz = x.clone()
        # B,N,C= x.shape
        if self.use_upsample=='up_or_down_ratio_score_2':
            x = process_point_cloud_mix(xyz,0.03,normal)
        elif self.use_upsample == 'median_hroup':
            I = Interpolation(0.03)
            x_t = xyz.transpose(2, 1)
            distance,idx_k = knn(x_t,k=20)
            # pts = I.random_k_neighbors_shape_invariant_perturb(xyz, None, normal)
            def expand_point_cloud_to_1024(original_points, accumulated_points, normal,distance,idx_k):
                if accumulated_points.shape[1] >= 1024:  
                    return accumulated_points
                pts_p = I.random_k_neighbors_shape_invariant_perturb(original_points, distance,idx_k, normal)   
                pts = torch.cat((accumulated_points, pts_p), dim=1) 
                return expand_point_cloud_to_1024(original_points, pts, normal,distance,idx_k)
            x = expand_point_cloud_to_1024(xyz,xyz,normal,distance,idx_k)
        # print("xxxxxxxx.shape:",x.shape)
        B,N,C= x.shape
        # print("self.sample_type:",self.sample_type)
        #add for sample_type
        if self.sample_type == 'fps':
            idx = pointnet2_utils.furthest_point_sample(x, npoint).long()
        elif self.sample_type == 'ffps_0.95':
            num1 = int(N * 0.95)
            # num2 = int(N * 0.95)
            weights,idx_k = cal_weight(x, k=20)
            # print("weights:",weights.shape)
            indices = torch.arange(N, device=weights.device).unsqueeze(0).repeat(B, 1)  
            # 将权重和索引堆叠起来，以便后续排序  
            weighted_indices = torch.stack([weights, indices], dim=-1)  
            # 根据权重对堆叠后的张量进行排序，获取排序后的索引  
            _, sorted_indices = torch.sort(weighted_indices[:, :, 0], dim=1,descending=True)  
            # 选择前num个 
            smallest_indices = sorted_indices[:, num1:]  
            # 创建一个与原weights形状相同的mask张量，初始化全为False  
            mask = torch.zeros(B, N, dtype=torch.bool, device=weights.device)  
            # 使用scatter_将smallest_indices对应的位置在mask中设置为True  
            mask.scatter_(1, smallest_indices, 1)  
            ###add for add
            # biggest_indices =  sorted_indices[:, :(N-num2)]
            # print("biggest_indices:",biggest_indices)
            # mask.scatter_(1, biggest_indices, 1)
            ###end
            # 将mask中标记为True的位置在weights中设置为-1
            weights[mask] = -1
            idx = pointnet2_utils.wfurthest_point_sample(x, weights, npoint).long()
        elif self.sample_type == 'wrs':
            idx,_ = weighted_random_point_sample(x, npoint, k=20,replace=True)
        else:
            idx = None
        # print("idx:",idx.shape)
        ###end
        ###add from models/pointnet.py
        x = x.transpose(2, 1).float()
        batch_size = x.size(0) # B,3,N
        # print("before x.shape:",x.shape)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x) #B,64,N,20
        # print("conv1 x.shape:",x.shape)
        x1 = x.max(dim=-1, keepdim=False)[0]#x1 :B,64,N
        # print("max x1.shape:",x1.shape)

        x = get_graph_feature(x1, k=self.k)
        # print("before conv2 x.shape:",x.shape)
        x = self.conv2(x)
        # print("after conv2 x.shape:",x.shape)
        
        x2 = x.max(dim=-1, keepdim=False)[0]
        # print("max x2.shape:",x2.shape)

        x = get_graph_feature(x2, k=self.k)
        # print("before conv3 x.shape:",x.shape)
        x = self.conv3(x)
        # print("after conv3 x.shape:",x.shape)
        x3 = x.max(dim=-1, keepdim=False)[0]
        # print("max x3.shape:",x3.shape)

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        # print("before conv4 x.shape:",x.shape)
        x4 = x.max(dim=-1, keepdim=False)[0]
        # print("conv1 x4.shape:",x4.shape)

        x = torch.cat((x1, x2, x3, x4), dim=1)#x:B,512,N
        # print("cat x:",x.shape)
        # print("self.conv5.weight.shape:",nn.Conv1d(512, 1024, kernel_size=1, bias=False).weight.shape)
        x = self.conv5(x)#x:B,1024,N，一维卷积是在最后维度上扫的
        # print("after conv5 x:",x.shape)

        if idx is not None:
            x_t = x.transpose(2, 1).float()##x_t:B,N,C
            new_x = index_points(x_t, idx)##new_x:B,npoint,C
            # print("new_x:",new_x.shape)
            new_x = new_x.transpose(2, 1).float()##new_x:B,C,npoint
            # print("new_x:",new_x.shape)
        else:
            new_x=x

        ###modify x->new_x
        x1 = F.adaptive_max_pool1d(new_x, 1).view(batch_size, -1)#x1:[B,1024]
        # print("after adaptive_max_pool1d x:",x1.shape)
        ###modify x->new_x
        x2 = F.adaptive_avg_pool1d(new_x, 1).view(batch_size, -1)#x2:[B,1024]
        # print("after adaptive_avg_pool1d x:",x2.shape)

        x = torch.cat((x1, x2), 1)
        # print("maxpool x.shape:",x.shape)
        if self.leaky_relu:
            act = lambda y: F.leaky_relu(y, negative_slope=0.2)
        else:
            act = F.relu

        x = act(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = act(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x
