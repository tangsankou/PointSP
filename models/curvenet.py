'''
Description: 
Autor: Jiachen Sun
Date: 2022-02-17 20:37:07
LastEditors: Jiachen Sun
LastEditTime: 2022-02-17 20:42:20
'''
import torch.nn as nn
import torch.nn.functional as F
from CurveNet.core.models.curvenet_cls import CurveNet as CurveNet_og
from all_utils import DATASET_NUM_CLASS

class CurveNet(nn.Module):

    # def __init__(self, task, dataset):
    def __init__(self, task, dataset,sample_type='fps',use_upsample='no'):

        super().__init__()
        self.task = task
        self.dataset = dataset
        ###modify
        sample_type = sample_type
        use_upsample=use_upsample
        if task == "cls":
            num_classes = DATASET_NUM_CLASS[dataset]
            # self.model = CurveNet_og(num_classes=num_classes)
            self.model = CurveNet_og(num_classes=num_classes,sample_type=sample_type,use_upsample=use_upsample)
        else:
            assert False

        

    def forward(self, **data):
        pc=data['pc']
        normal = data.get('normal')
        cls = data.get('cls')  # 默认为 None 如果 'cls' 键不存在
        pc = pc.to(next(self.parameters()).device)
        # pc = pc.permute(0, 2, 1).contiguous()
        if self.task == 'cls':
            assert cls is None
            #logit = self.model(pc)
            logit = self.model(pc,normal)
            out = {'logit': logit}
        else:
            assert False

        return out
