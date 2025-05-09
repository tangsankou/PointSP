"""
Modified by 
@Author: Pin Tang
@Contact: tangpin1874@163.com
@Time: 2025/05/6 21:10 PM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import sample_and_group
from .interpolation import Interpolation
from .sampling import process_point_cloud_mix,knn

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)   
        x = x.reshape(-1, d, s) 
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class Pct(nn.Module):
    def __init__(self, args, output_channels=40):
        super(Pct, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(1024),
                                    nn.LeakyReLU(negative_slope=0.2))


        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x,normal):
        xyz=x
        B,N,C=xyz.shape
        ###PointSP
        if self.args.use_upsample=='lgp_or_lgd':
            xyz = process_point_cloud_mix(xyz,0.03,normal)
        elif self.args.use_upsample == 'clgp':
            I = Interpolation(0.03)
            distance,idx_k = knn(xyz,k=20)
            # pts = I.random_k_neighbors_shape_invariant_perturb(xyz, None, normal)
            def expand_point_cloud_to_1024(original_points, accumulated_points, normal,distance,idx_k):
                if accumulated_points.shape[1] >= 1024:  
                    return accumulated_points[:,:1024,:]
                pts_p = I.random_k_neighbors_shape_invariant_perturb(original_points, distance,idx_k, normal)   
                pts = torch.cat((accumulated_points, pts_p), dim=1) 
                return expand_point_cloud_to_1024(original_points, pts, normal,distance,idx_k)
            xyz = expand_point_cloud_to_1024(xyz.clone(),xyz.clone(),normal,distance,idx_k)
        ###end
        
        x = xyz.permute(0, 2, 1)
        batch_size, C, N = x.size()
        # print("N:",int(N/2))
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
                
        ###origin
        # new_xyz, new_feature = sample_and_group(npoint=int(N/2), radius=0.15, nsample=32, xyz=xyz, points=x,sample_type = self.args.sample_type)         

        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x,sample_type = self.args.sample_type)         
        feature_0 = self.gather_local_0(new_feature)
        feature = feature_0.permute(0, 2, 1)
        ###origin
        # new_xyz, new_feature = sample_and_group(npoint=int(N/4), radius=0.2, nsample=32, xyz=new_xyz, points=feature,sample_type = 'fps') 
        
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature,sample_type = 'fps') 
        feature_1 = self.gather_local_1(new_feature)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x

class Point_Transformer_Last(nn.Module):
    def __init__(self, args, channels=256):
        super(Point_Transformer_Last, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.sa1 = SA_Layer(channels)
        self.sa2 = SA_Layer(channels)
        self.sa3 = SA_Layer(channels)
        self.sa4 = SA_Layer(channels)

    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()

        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        return x

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
    

###RPC
from .GDANet_cls import GDM, local_operator, SGCAM
# from .interpolation import Interpolation
from .sampling import weighted_random_point_sample,cal_weight
from pointnet2_ops import pointnet2_utils
from .util import index_points
class RPC(nn.Module):
    def __init__(self, args, output_channels=40):
        super(RPC, self).__init__()
        self.args = args

        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.bn11 = nn.BatchNorm2d(128, momentum=0.1)
        self.bn12 = nn.BatchNorm1d(256, momentum=0.1)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=True),
                                   self.bn1)
        self.conv11 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=True),
                                    self.bn11)
        self.SGCAM_1s = SGCAM(128)
        self.SGCAM_1g = SGCAM(128)

        self.pt_last = Point_Transformer_Last(args)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1280, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)


    def forward(self, x,normal):
        
        batch_size, _, _ = x.size()
        ###pointdr
        npoint=512
        xyz=x.clone()
        B,N,C=xyz.shape

        
        ###上采样random_k，下采样score
        if self.args.use_upsample=='up_or_down_ratio_score_2':
            x = process_point_cloud_mix(xyz,0.03,normal)
        elif self.args.use_upsample=='half_roup':
            if N<1024:
                I = Interpolation(0.03)
                d, idx_k = knn(xyz, k=20)
                newx = I.random_k_neighbors_shape_invariant_perturb(xyz,d ,idx_k, normal)
                num_points_to_select = int(1024-N)
                # sampled_indices = torch.randperm(N)[:num_points_to_select]
                x = torch.cat((xyz, newx[:, :num_points_to_select]), dim=1)
            else:
                x = xyz.contiguous()
        elif self.args.use_upsample == 'median_hroup':
            I = Interpolation(0.03)
            distance,idx_k = knn(xyz,k=20)
            # pts = I.random_k_neighbors_shape_invariant_perturb(xyz, None, normal)
            def expand_point_cloud_to_1024(original_points, accumulated_points, normal,distance,idx_k):
                if accumulated_points.shape[1] >= 1024:  
                    return accumulated_points[:,:1024,:]
                pts_p = I.random_k_neighbors_shape_invariant_perturb(original_points, distance,idx_k, normal)   
                pts = torch.cat((accumulated_points, pts_p), dim=1) 
                return expand_point_cloud_to_1024(original_points, pts, normal,distance,idx_k)
            x = expand_point_cloud_to_1024(xyz.clone(),xyz.clone(),normal,distance,idx_k)

        
        # print("after upsample:",x.shape)
        B,N,C=x.shape
        # #add for sample_type
        if self.args.sample_type == 'fps':
            idx = pointnet2_utils.furthest_point_sample(x, npoint).long()
        elif self.args.sample_type == 'ffps_0.95':
            num1 = int(N * 0.95)
            # num2 = int(N * 0.95)
            weights,idx_k = cal_weight(x, k=20)
            # print("weights:",weights.shape)
            indices = torch.arange(N, device=weights.device).unsqueeze(0).repeat(B, 1)
            # print("indices:",indices.shape) 
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
            idx = pointnet2_utils.wfurthest_point_sample(x.contiguous(), weights, npoint).long()
        elif self.args.sample_type == 'wrs':
            idx,_ = weighted_random_point_sample(x, npoint, k=20,replace=True)
        else:
            idx = None
        x = x.permute(0, 2, 1)
        # print("x.shape:",x)

        x1 = local_operator(x, k=30) #x:B,C,N

        # print("x11.shape:",x1)
        x1 = F.relu(self.conv1(x1))
        # print("x12.shape:",x1)
        x1 = F.relu(self.conv11(x1))
        # print("x13.shape:",x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]
        # print("x14.shape:",x1)

        # Geometry-Disentangle Module:
        x1s, x1g = GDM(x1, M=256)

        # Sharp-Gentle Complementary Attention Module:
        y1s = self.SGCAM_1s(x1, x1s.transpose(2, 1))
        y1g = self.SGCAM_1g(x1, x1g.transpose(2, 1))
        feature_1 = torch.cat([y1s, y1g], 1)

        x = self.pt_last(feature_1)
        x = torch.cat([x, feature_1], dim=1)
        x = self.conv_fuse(x)

        ###before
        ##maxpooling add idx
        # print("xyz.shape:",x)#x:b,c,n
        # print("idx:",idx)
        if idx is not None:
            x_t = x.transpose(2, 1).float()##x_t:B,N,C
            # print("x_t:",x_t)
            new_x = index_points(x_t, idx)##new_x:B,npoint,C
            # print("new_x:",new_x)
            new_x = new_x.transpose(2, 1).float()##new_x左:B,C,npoint
            # print("new_x:",new_x.shape)

        else:
            new_x=x
        ###end
        # print("new_x:",new_x)

        ###change x to new_x
        x = F.adaptive_max_pool1d(new_x, 1).view(batch_size, -1)
        # print("x maxpool:",x)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        # print("x mlp1:",x)

        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        # print("x mlp3:",x)

        return x