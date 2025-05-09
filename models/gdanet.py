'''
Description: 
Autor: Jiachen Sun
Date: 2022-02-22 23:22:17
LastEditors: Jiachen Sun
LastEditTime: 2022-02-23 00:16:25
'''
import torch
import torch.nn as nn
from GDANet.model.GDANet_cls import GDANET as GDANET_og
from all_utils import DATASET_NUM_CLASS

class GDANET(nn.Module):

    # def __init__(self, task, dataset):
    def __init__(self, task,dataset,sample_type='no',use_upsample='no'):
    
        super().__init__()
        self.task =  task
        num_class = DATASET_NUM_CLASS[dataset]
        ###modify
        sample_type = sample_type
        use_upsample=use_upsample
        if task == 'cls':
            # self.model = GDANET_og(number_class=num_class)
            self.model = GDANET_og(number_class=num_class,sample_type=sample_type,use_upsample=use_upsample)

        else:
            assert False

    def forward(self, **data):
        pc=data['pc']
        normal = data.get('normal')
        cls = data.get('cls')  # 默认为 None 如果 'cls' 键不存在
        # batch_size = pc.shape[0]
        # pc=pc.permute(0,2,1).contiguous()
        pc = pc.to(next(self.parameters()).device)
        if self.task == 'cls':
            assert cls is None
            # assert normal is None
            #logit = self.model(pc)
            logit = self.model(pc,normal)
            out = {'logit': logit}
        else:
            assert False
        return out
