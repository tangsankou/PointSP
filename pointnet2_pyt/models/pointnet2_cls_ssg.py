"""
Modified by 
@Author: Pin Tang
@Contact: tangpin1874@163.com
@Time: 2025/05/6 21:45 PM
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from pointnet2_pyt.models.pointnet2_utils import PointNetSetAbstraction
from PCT_Pytorch.interpolation import Interpolation
from PCT_Pytorch.sampling import knn,process_point_cloud_mix

class Pointnet2SSG(nn.Module):
    # def __init__(self,args,normal_channel=True):
        # super(Pointnet2SSG, self).__init__()
    def __init__(self, num_classes, input_channels=3, sample_type='fps',use_upsample='no'):
        super(Pointnet2SSG, self).__init__()

        in_channel = 3
        num_class = num_classes
        self.sample_type = sample_type
        self.use_upsample = use_upsample
        # self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False,sample_type=self.sample_type)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False,sample_type=self.sample_type)###????
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True,sample_type=self.sample_type)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz,normal):
        # if self.normal_channel:
        #     norm = xyz[:, 3:, :]
        #     xyz = xyz[:, :3, :]
        # else:
        norm = None
        B, N, C = xyz.shape
        if self.use_upsample=='lgp_or_lgd':
            xyz = process_point_cloud_mix(xyz,0.03,normal)
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
            xyz = expand_point_cloud_to_1024(xyz.clone(),xyz.clone(),normal,distance,idx_k)
        else:
            xyz = xyz.contiguous()
        # print("after:",xyz.shape)
        xyz = xyz.transpose(2, 1)
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x, l3_points



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
