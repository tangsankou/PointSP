
import torch.nn as nn
import torch.nn.functional as F
from dgcnn.pytorch.model import DGCNN as DGCNN_original
from all_utils import DATASET_NUM_CLASS

class DGCNN(nn.Module):

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
                    self.k = 20
                    self.emb_dims = 1024
                    self.dropout = 0.5
                    self.leaky_relu = 1
                    self.sample_type = sample_type  # 添加 sample_type 属性
                    self.use_upsample = use_upsample
            args = Args(sample_type=self.sample_type, use_upsample=self.use_upsample)  # 传递 sample_type
            # args = Args()
            self.model = DGCNN_original(args, output_channels=num_classes)
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
