"""
Modified by 
@Author: Pin Tang
@Contact: tangpin1874@163.com
@Time: 2025/05/6 21:10 PM
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from PCT_Pytorch.interpolation import Interpolation
from PCT_Pytorch.sampling import knn,weighted_random_point_sample,process_point_cloud_mix,weighted_random_point_sample,cal_weight
from pointnet2_ops import pointnet2_utils
from PCT_Pytorch.util import index_points

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    ###add idx
    def forward(self, x,idx):
        n_pts = x.size()[2]##x:[B,C,N]
        trans = self.stn(x)
        x = x.transpose(2, 1)##left x:[B,N,C]
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)##left x:[B,C,N]
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        ###maxpooling add idx
        # print("xyz.shape:",x.shape)
        # print("idx:",idx.shape)
        if idx is not None:
            x_t = x.transpose(2, 1).float()##x_t:B,N,C
            new_x = index_points(x_t, idx)##new_x:B,npoint,C
            # print("new_x:",new_x.shape)
            new_x = new_x.transpose(2, 1).float()##new_x:B,C,npoint
            # print("new_x:",new_x.shape)

        else:
            new_x=x
        ###end
        # print("trans_feat:",trans_feat.shape)
        x = torch.max(new_x, 2, keepdim=True)[0]###change the second x to new_x
        # print("x:",x.shape)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    ###before
    # def __init__(self, k=2, feature_transform=False):
    def __init__(self, k=2, feature_transform=False, sample_type='no',use_upsample='no'):
        # print("sample:",sample_type)
        # print("use:",use_upsample)
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        ###add 
        self.sample_type = sample_type
        self.use_upsample = use_upsample

    ###add normal
    def forward(self, x,normal):
        #add for use_upsample
        npoint=512
        xyz = x.clone()
        # B,N,C= x.shape
        ###PointSP
        if self.use_upsample=='lgp_or_lgd':
            x = process_point_cloud_mix(xyz,0.03,normal)
        elif self.use_upsample == 'clgp':
            I = Interpolation(0.03)
            distance,idx_k = knn(xyz,k=20)
            # pts = I.random_k_neighbors_shape_invariant_perturb(xyz, None, normal)
            def expand_point_cloud_to_1024(original_points, accumulated_points, normal,distance,idx_k):
                if accumulated_points.shape[1] >= 1024:  
                    return accumulated_points
                pts_p = I.random_k_neighbors_shape_invariant_perturb(original_points, distance,idx_k, normal)   
                pts = torch.cat((accumulated_points, pts_p), dim=1) 
                return expand_point_cloud_to_1024(original_points, pts, normal,distance,idx_k)
            x = expand_point_cloud_to_1024(xyz,xyz,normal,distance,idx_k)
        # print("x.shape:",x.shape)
        B,N,C= x.shape

        #add for sample_type
        if self.sample_type == 'fps':
            idx = pointnet2_utils.furthest_point_sample(x, npoint).long()
        elif self.sample_type == 'ffps':
            num1 = int(N * 0.95)
            # num2 = int(N * 0.95)
            weights,idx_k = cal_weight(x, k=20)
            indices = torch.arange(N, device=weights.device).unsqueeze(0).repeat(B, 1)
            weighted_indices = torch.stack([weights, indices], dim=-1)  
            _, sorted_indices = torch.sort(weighted_indices[:, :, 0], dim=1,descending=True)
            smallest_indices = sorted_indices[:, num1:]    
            mask = torch.zeros(B, N, dtype=torch.bool, device=weights.device) 
            mask.scatter_(1, smallest_indices, 1)  
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

        x, trans, trans_feat = self.feat(x,idx)###add idx
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2), p=2))
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())
