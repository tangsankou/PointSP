'''
Description: 
Autor: Jiachen Sun
Date: 2022-02-17 20:50:58
LastEditors: Jiachen Sun
LastEditTime: 2022-02-21 21:18:02
'''
import torch.nn as nn
from pointMLP.classification_ModelNet40.models.pointmlp import pointMLP as pointMLP_original
from all_utils import DATASET_NUM_CLASS


class pointMLP(nn.Module):

    def __init__(self, task, dataset):
        super().__init__()
        self.task = task
        self.dataset = dataset

        if task == "cls":
            num_classes = DATASET_NUM_CLASS[dataset]
            self.model = pointMLP_original(num_classes=num_classes)

        else:
            assert False

    def forward(self, **data):
        pc=data['pc']
        normal = data.get('normal')
        cls = data.get('cls')  # 默认为 None 如果 'cls' 键不存在
        pc = pc.to(next(self.parameters()).device)
        pc = pc.permute(0, 2, 1).contiguous()
        if self.task == 'cls':
            assert cls is None
            logit = self.model(pc)
            out = {'logit': logit}
        else:
            assert False

        return out
