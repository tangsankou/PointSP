# based on: https://github.com/fxia22/pointnet.pytorch/blob/master/utils/train_classification.py
import torch.nn as nn
from pointnet_pyt.pointnet.model import PointNetCls
from all_utils import DATASET_NUM_CLASS

class PointNet(nn.Module):
    ###before
    # def __init__(self, dataset, task):
    def __init__(self, dataset, task,sample_type='no',use_upsample='no'):
        super().__init__()
        self.task = task
        num_class = DATASET_NUM_CLASS[dataset]
        ###modify
        sample_type = sample_type
        use_upsample=use_upsample
        if self.task == 'cls_trans':
            self.model =  PointNetCls(k=num_class, feature_transform=True,sample_type=sample_type,use_upsample=use_upsample)
        else:
            assert False

    def forward(self, **data):
        pc=data['pc']
        normal = data.get('normal')
        cls = data.get('cls')  # 默认为 None 如果 'cls' 键不存在
        # pc = pc.to(next(self.parameters()).device)
        pc = pc.cuda()
        # pc = pc.transpose(2, 1).float()
        if self.task == 'cls_trans':
            # logit, _, trans_feat = self.model(pc)
            logit, _, trans_feat = self.model(pc,normal)
        else:
            assert False

        out = {'logit': logit, 'trans_feat': trans_feat}
        return out
