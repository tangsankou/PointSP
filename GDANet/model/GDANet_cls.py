"""
Modified by 
@Author: Pin Tang
@Contact: tangpin1874@163.com
@Time: 2025/05/6 21:42 PM
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from .util.GDANet_util import local_operator, GDM, SGCAM
from PCT_Pytorch.interpolation import Interpolation
from PCT_Pytorch.sampling import knn,weighted_random_point_sample,process_point_cloud_mix,weighted_random_point_sample,cal_weight
from pointnet2_ops import pointnet2_utils
from PCT_Pytorch.util import index_points

class GDANET(nn.Module):
    # def __init__(self, number_class=40):
    def __init__(self, number_class=40,sample_type='no',use_upsample='no'):
    
        super(GDANET, self).__init__()

        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn11 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn12 = nn.BatchNorm1d(64, momentum=0.1)

        self.bn2 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn21 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn22 = nn.BatchNorm1d(64, momentum=0.1)

        self.bn3 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn31 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn32 = nn.BatchNorm1d(128, momentum=0.1)

        self.bn4 = nn.BatchNorm1d(512, momentum=0.1)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=True),
                                   self.bn1)
        self.conv11 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),
                                    self.bn11)
        self.conv12 = nn.Sequential(nn.Conv1d(64 * 2, 64, kernel_size=1, bias=True),
                                    self.bn12)

        self.conv2 = nn.Sequential(nn.Conv2d(67 * 2, 64, kernel_size=1, bias=True),
                                   self.bn2)
        self.conv21 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=True),
                                    self.bn21)
        self.conv22 = nn.Sequential(nn.Conv1d(64 * 2, 64, kernel_size=1, bias=True),
                                    self.bn22)

        self.conv3 = nn.Sequential(nn.Conv2d(131 * 2, 128, kernel_size=1, bias=True),
                                   self.bn3)
        self.conv31 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=True),
                                    self.bn31)
        self.conv32 = nn.Sequential(nn.Conv1d(128, 128, kernel_size=1, bias=True),
                                    self.bn32)

        self.conv4 = nn.Sequential(nn.Conv1d(256, 512, kernel_size=1, bias=True),
                                   self.bn4)

        self.SGCAM_1s = SGCAM(64)
        self.SGCAM_1g = SGCAM(64)
        self.SGCAM_2s = SGCAM(64)
        self.SGCAM_2g = SGCAM(64)

        self.linear1 = nn.Linear(1024, 512, bias=True)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.4)
        self.linear2 = nn.Linear(512, 256, bias=True)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.4)
        self.linear3 = nn.Linear(256, number_class, bias=True)

        ###add 
        self.sample_type = sample_type
        self.use_upsample = use_upsample
    def forward(self, x,normal):
        # print("x.shape:",x.shape)
        npoint=512
        xyz = x.clone()
        # B,N,C= x.shape
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
        # print("after up x.shape:",x.shape)
        B,N,C= x.shape

        #add for sample_type
        if self.sample_type == 'fps':
            idx = pointnet2_utils.furthest_point_sample(x, npoint).long()
        elif self.sample_type == 'ffps':
            num1 = int(N * 0.95)
            weights,idx_k = cal_weight(x, k=20)
            indices = torch.arange(N, device=weights.device).unsqueeze(0).repeat(B, 1) 
            weighted_indices = torch.stack([weights, indices], dim=-1)  
            _, sorted_indices = torch.sort(weighted_indices[:, :, 0], dim=1,descending=True)  
            # 选择前num个 
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
        ###add from models/gdanet.py
        x=x.permute(0,2,1).contiguous()
        # print("x.shape...",x.shape)
        B, C, N = x.size()
        ###############
        """block 1"""
        # Local operator:
        x1 = local_operator(x, k=30)
        # print("x1:",x1.shape)
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv11(x1))
        # print("x1:",x1.shape)

        x1 = x1.max(dim=-1, keepdim=False)[0]
        # print("x1:",x1.shape)

        # Geometry-Disentangle Module:
        x1s, x1g = GDM(x1, M=256)
        # print("x1s:",x1s.shape)

        # Sharp-Gentle Complementary Attention Module:
        y1s = self.SGCAM_1s(x1, x1s.transpose(2, 1))
        y1g = self.SGCAM_1g(x1, x1g.transpose(2, 1))
        # print("y1s:",y1s.shape)
        z1 = torch.cat([y1s, y1g], 1)
        # print("z1::",z1.shape)
        z1 = F.relu(self.conv12(z1))
        # print("z1::",z1.shape)

        ###############
        """block 2"""
        x1t = torch.cat((x, z1), dim=1)
        # print("x1t:",x1t.shape)

        x2 = local_operator(x1t, k=30)
        # print("x2local:",x2.shape)
        x2 = F.relu(self.conv2(x2))
        x2 = F.relu(self.conv21(x2))

        
        x2 = x2.max(dim=-1, keepdim=False)[0]###change the second x to new_x
        # print("x2.shape:",x2.shape)
        x2s, x2g = GDM(x2, M=256)
        # print("x2s:",x2s.shape)

        y2s = self.SGCAM_2s(x2, x2s.transpose(2, 1))
        # print("y2s:",y2s.shape)
        y2g = self.SGCAM_2g(x2, x2g.transpose(2, 1))
        z2 = torch.cat([y2s, y2g], 1)
        # print("z2:",z2.shape)

        z2 = F.relu(self.conv22(z2))
        # print("z2:",z2.shape)
        ###############
        x2t = torch.cat((x1t, z2), dim=1)
        # print("x2t:",x2t.shape)
        x3 = local_operator(x2t, k=30)
        # print("x3:",x3.shape)
        x3 = F.relu(self.conv3(x3))
        x3 = F.relu(self.conv31(x3))
        # print("x33.shape:",x3.shape)
        x3 = x3.max(dim=-1, keepdim=False)[0]
        # print("x33 after max.shape:",x3.shape)

        z3 = F.relu(self.conv32(x3))
        # print("z3:",z3.shape)

        ###############
        x = torch.cat((z1, z2, z3), dim=1)
        # print("x:x:",x.shape)
        x = F.relu(self.conv4(x))
        # print("x:x:mlp",x.shape)

        ###before
        ###maxpooling add idx
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
        ###modify x->new_x
        x11 = F.adaptive_max_pool1d(new_x, 1).view(B, -1)
        ###modify x->new_x
        x22 = F.adaptive_avg_pool1d(new_x, 1).view(B, -1)

        x = torch.cat((x11, x22), 1)
        # print("x cat:",x.shape)

        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)

        return x
