import torch.nn as nn
import torch.nn.functional as F
import torch
from pointnet2_pyt.models.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from PCT_Pytorch.interpolation import Interpolation
from PCT_Pytorch.sampling import knn,weighted_random_point_sample,process_point_cloud

class Pointnet2MSG(nn.Module):
    # def __init__(self,args,normal_channel=True):
        # super(Pointnet2SSG, self).__init__()
    def __init__(self, num_classes, input_channels=3, sample_type='fps',use_upsample='no'):
        super(Pointnet2MSG, self).__init__()
# class get_model(nn.Module):
#     def __init__(self,args,normal_channel=True):
#         super(get_model, self).__init__()
        in_channel = input_channels
        num_class = num_classes
        self.sample_type = sample_type
        self.use_upsample=use_upsample
        # self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]],sample_type=self.sample_type)
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]],sample_type=self.sample_type)
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True,sample_type=self.sample_type)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz,normal):
        B, N, C = xyz.shape
        # if self.normal_channel:
        #     norm = xyz[:, 3:, :]
        #     xyz = xyz[:, :3, :]
        # else:
        norm = None
        if self.use_upsample=='up_or_down':
            xyz = process_point_cloud(xyz, 0.03,normal)
        elif self.use_upsample=='r_rwup':
            I = Interpolation(0.03)
            ###wrs sample replace=True perb
            centroids,idx_k = weighted_random_point_sample(xyz, N, k=20,replace = True) #(B,N)
            xyz_w = torch.gather(xyz, 1, centroids.unsqueeze(-1).expand(B, N, C))
            newx = I.random_k_neighbors_shape_invariant_perturb(xyz_w, idx_k, normal)
            xyz = torch.cat((xyz, newx), dim=1)
        elif self.use_upsample=='r_oup':
            I = Interpolation(0.03)
            _, idx_k = knn(xyz, k=20)
            newx = I.random_k_neighbors_shape_invariant_perturb(xyz, idx_k, normal)
            xyz = torch.cat((xyz, newx), dim=1)
        else:
            xyz = xyz.contiguous()
        
        xyz = xyz.transpose(2, 1)

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)


        return x,l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


