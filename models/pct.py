import torch.nn as nn
from PCT_Pytorch.model import Pct as Pct_original
from all_utils import DATASET_NUM_CLASS
###modify: add sample_type, use_upsample
class Pct(nn.Module):

    def __init__(self, task, dataset,sample_type,use_upsample):
        super().__init__()
        self.task = task
        self.dataset = dataset
        ###modify
        self.sample_type = sample_type
        self.use_upsample = use_upsample

        if task == "cls":
            num_classes = DATASET_NUM_CLASS[dataset]
            # default arguments
            class Args:
                def __init__(self,sample_type,use_upsample):
                    self.dropout = 0.5
                    self.sample_type = sample_type  # 添加 sample_type 属性
                    self.use_upsample = use_upsample
            args = Args(sample_type=self.sample_type, use_upsample=self.use_upsample)  # 传递 sample_type
                
            # args = Args()
            self.model = Pct_original(args, output_channels=num_classes)

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
            logit = self.model(pc,normal)
            out = {'logit': logit}
        else:
            assert False

        return out
