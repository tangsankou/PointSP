'''
Description: 
Autor: Jiachen Sun
Date: 2022-02-16 22:23:16
LastEditors: Jiachen Sun
LastEditTime: 2022-02-24 22:36:59
'''
import torch
import torch.nn as nn
from pointnet2_pyt.models.pointnet2_cls_msg import Pointnet2MSG
from pointnet2_pyt.models.pointnet2_cls_ssg import Pointnet2SSG
from all_utils import DATASET_NUM_CLASS

class PointNet2(nn.Module):

    def __init__(self, task, dataset,sample_type,use_upsample):
        super().__init__()
        self.task =  task
        ###modify
        sample_type = sample_type
        use_upsample=use_upsample
        num_class = DATASET_NUM_CLASS[dataset]
        if task == 'cls':
            ###origin

            # self.model = Pointnet2MSG(num_classes=num_class, input_channels=0, sample_type = sample_type)
            ###modify add sample_type
            self.model = Pointnet2SSG(num_classes=num_class, input_channels=3, sample_type = sample_type,use_upsample=use_upsample)

        else:
            assert False

    def forward(self, **data):
        # pc = pc.to(next(self.parameters()).device)
        pc=data['pc']
        normal = data.get('normal')
        cls = data.get('cls')  # 默认为 None 如果 'cls' 键不存在
        pc = pc.cuda()
        # print("pc.shape:",pc.shape)
        if self.task == 'cls':
            assert cls is None
            # assert normal is None
            logit,trans = self.model(pc,normal)
            out = {'logit': logit}
        else:
            assert False
        return out
