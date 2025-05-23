"""
Modified by 
@Author: Pin Tang
@Contact: tangpin1874@163.com
@Time: 2024/11/5 21:45 PM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import random
from dataloader import create_dataloader
from time import time
from timer import ProfileTimer
from progressbar import ProgressBar
import models
from collections import defaultdict
import os
import re
import numpy as np
import argparse
from all_utils import (
    TensorboardManager, PerfTrackTrain,
    PerfTrackVal,PerfTrackValomni, TrackTrain, smooth_loss, DATASET_NUM_CLASS,
    rscnn_voting_evaluate_cls, pn2_vote_evaluate_cls)
from configs import get_cfg_defaults
import pprint
from pointnet_pyt.pointnet.model import feature_transform_regularizer
import sys
import aug_utils
from third_party import bn_helper, tent_helper

from PCT_Pytorch.sampling import get_normal_vector

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cpu':
    print('WARNING: Using CPU')

def adapt_bn(data,model,cfg):
    model = bn_helper.configure_model(model,eps=1e-5, momentum=0.1,reset_stats=False,no_stats=False)
    for _ in range(cfg.ITER):
        model(**data) 
    print("Adaptation Done ...")
    model.eval()
    return model

def adapt_tent(data,model,cfg):
    model = tent_helper.configure_model(model,eps=1e-5, momentum=0.1)
    parameter,_ = tent_helper.collect_params(model)
    optimizer_tent = torch.optim.SGD(parameter, lr=0.001,momentum=0.9)

    for _ in range(cfg.ITER):
        # index = np.random.choice(args.number,args.batch_size,replace=False)
        tent_helper.forward_and_adapt(data,model,optimizer_tent)
    print("Adaptation Done ...")
    model.eval()
    return model


def check_inp_fmt(task, data_batch, dataset_name):
    if task in ['cls', 'cls_trans']:
        # assert set(data_batch.keys()) == {'pc', 'label'}
        # print(data_batch['pc'],data_batch['label'])
        pc, label = data_batch['pc'], data_batch['label']
        # special case made for modelnet40_dgcnn to match the
        # original implementation
        # dgcnn loads torch.DoubleTensor for the test dataset
        if dataset_name == 'modelnet40_dgcnn':
            assert isinstance(pc, torch.FloatTensor if DEVICE.type == 'cpu' else torch.cuda.FloatTensor) or isinstance(
                pc, torch.DoubleTensor if DEVICE.type == 'cpu' else torch.cuda.DoubleTensor)
        else:
            assert isinstance(pc, torch.FloatTensor if DEVICE.type == 'cpu' else torch.cuda.FloatTensor)
        assert isinstance(label, torch.LongTensor if DEVICE.type == 'cpu' else torch.cuda.LongTensor)
        assert len(pc.shape) == 3
        assert len(label.shape) == 1
        b1, _, y = pc.shape[0], pc.shape[1], pc.shape[2]
        b2 = label.shape[0]
        assert b1 == b2
        assert y == 3
        assert label.max().item() < DATASET_NUM_CLASS[dataset_name]
        assert label.min().item() >= 0
    else:
        assert NotImplemented


def check_out_fmt(task, out, dataset_name):
    if task == 'cls':
        assert set(out.keys()) == {'logit'}
        logit = out['logit']
        assert isinstance(logit, torch.FloatTensor if DEVICE.type == 'cpu' else torch.cuda.FloatTensor)
        assert len(logit.shape) == 2
        assert DATASET_NUM_CLASS[dataset_name] == logit.shape[1]
    elif task == 'cls_trans':
        assert set(out.keys()) == {'logit', 'trans_feat'}
        logit = out['logit']
        trans_feat = out['trans_feat']
        assert isinstance(logit, torch.FloatTensor if DEVICE.type == 'cpu' else torch.cuda.FloatTensor)
        assert isinstance(trans_feat, torch.FloatTensor if DEVICE.type == 'cpu' else torch.cuda.FloatTensor)
        assert len(logit.shape) == 2
        assert len(trans_feat.shape) == 3
        assert trans_feat.shape[0] == logit.shape[0]
        # 64 coming from pointnet implementation
        assert (trans_feat.shape[1] == trans_feat.shape[2]) and (trans_feat.shape[1] == 64)
        assert DATASET_NUM_CLASS[dataset_name] == logit.shape[1]
    else:
        assert NotImplemented

#根据任务类型和模型类型，从data_batch中获取输入数据，并返回一个字典作为输入。
def get_inp(task, model, data_batch, batch_proc, dataset_name,model_name):
    check_inp_fmt(task, data_batch, dataset_name)
    if not batch_proc is None:
        data_batch = batch_proc(data_batch, DEVICE)
        check_inp_fmt(task, data_batch, dataset_name)

    if isinstance(model, nn.DataParallel):
        model_type = type(model.module)
    else:
        model_type = type(model)
    ###modify for up
    
    if model_name in ['pct','pointnet2','pointnet','dgcnn','gdanet','curvenet','rpc']:
        points = data_batch['pc']
        # print("points:",points.shape)
        device = points.device
        normal = torch.zeros_like(points).to(device)
        for i in range(points.shape[0]):
            normal[i] = get_normal_vector(points[i].unsqueeze(0)).squeeze(0)
        # print("normal:",normal.shape)
        inp = {'pc': points,'normal':normal}

    elif task in ['cls', 'cls_trans']:
        pc = data_batch['pc']
        inp = {'pc': pc}
    else:
        assert False

    return  inp


def get_loss(task, loss_name, data_batch, out, dataset_name):
    """
    Returns the tensor loss function
    :param task:
    :param loss_name:
    :param data_batch: batched data; note not applied data_batch
    :param out: output from the model
    :param dataset_name:
    :return: tensor
    """
    check_out_fmt(task, out, dataset_name)
    if task == 'cls':
        label = data_batch['label'].to(out['logit'].device)
        if loss_name == 'cross_entropy':
            if 'label_2' in data_batch.keys():
                label_2 = data_batch['label_2'].to(out['logit'].device)
                if isinstance(data_batch['lam'],torch.Tensor):
                    loss = 0
                    for i in range(data_batch['pc'].shape[0]):
                        loss_tmp = smooth_loss(out['logit'][i].unsqueeze(0), label[i].unsqueeze(0).long()) * (1 - data_batch['lam'][i]) + smooth_loss(out['logit'][i].unsqueeze(0), label_2[i].unsqueeze(0).long()) * data_batch['lam'][i]
                        loss += loss_tmp
                    loss = loss / data_batch['pc'].shape[0]
                else:
                    loss = smooth_loss(out['logit'], label) * (1 - data_batch['lam']) + smooth_loss(out['logit'], label_2) * data_batch['lam']
            else:
                loss = F.cross_entropy(out['logit'], label)
        # source: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/util.py
        elif loss_name == 'smooth':
            if 'label_2' in data_batch.keys():
                label_2 = data_batch['label_2'].to(out['logit'].device)
                if isinstance(data_batch['lam'],torch.Tensor):
                    loss = 0
                    for i in range(data_batch['pc'].shape[0]):
                        loss_tmp = smooth_loss(out['logit'][i].unsqueeze(0), label[i].unsqueeze(0).long()) * (1 - data_batch['lam'][i]) + smooth_loss(out['logit'][i].unsqueeze(0), label_2[i].unsqueeze(0).long()) * data_batch['lam'][i]
                        loss += loss_tmp
                    loss = loss / data_batch['pc'].shape[0]
                else:
                    loss = smooth_loss(out['logit'], label) * (1 - data_batch['lam']) + smooth_loss(out['logit'], label_2) * data_batch['lam']
            else:
                loss = smooth_loss(out['logit'], label)
        else:
            assert False
    elif task == 'cls_trans':
        label = data_batch['label'].to(out['logit'].device)
        trans_feat = out['trans_feat']
        logit = out['logit']
        if loss_name == 'cross_entropy':
            if 'label_2' in data_batch.keys():
                label_2 = data_batch['label_2'].to(out['logit'].device)
                if isinstance(data_batch['lam'],torch.Tensor):
                    loss = 0
                    for i in range(data_batch['pc'].shape[0]):
                        loss_tmp = smooth_loss(out['logit'][i].unsqueeze(0), label[i].unsqueeze(0).long()) * (1 - data_batch['lam'][i]) + smooth_loss(out['logit'][i].unsqueeze(0), label_2[i].unsqueeze(0).long()) * data_batch['lam'][i]
                        loss += loss_tmp
                    loss = loss / data_batch['pc'].shape[0]
                else:
                    loss = smooth_loss(out['logit'], label) * (1 - data_batch['lam']) + smooth_loss(out['logit'], label_2) * data_batch['lam']
            else:
                loss = F.cross_entropy(out['logit'], label)
            loss += feature_transform_regularizer(trans_feat) * 0.001
        elif loss_name == 'smooth':
            if 'label_2' in data_batch.keys():
                label_2 = data_batch['label_2'].to(out['logit'].device)
                if isinstance(data_batch['lam'],torch.Tensor):
                    loss = 0
                    for i in range(data_batch['pc'].shape[0]):
                        loss_tmp = smooth_loss(out['logit'][i].unsqueeze(0), label[i].unsqueeze(0).long()) * (1 - data_batch['lam'][i]) + smooth_loss(out['logit'][i].unsqueeze(0), label_2[i].unsqueeze(0).long()) * data_batch['lam'][i]
                        loss += loss_tmp
                    loss = loss / data_batch['pc'].shape[0]
                else:
                    loss = smooth_loss(out['logit'], label) * (1 - data_batch['lam']) + smooth_loss(out['logit'], label_2) * data_batch['lam']
            else:
                loss = smooth_loss(out['logit'], label)
            loss += feature_transform_regularizer(trans_feat) * 0.001
        else:
            assert False
    else:
        assert False

    return loss

###modify add use_upsample
def validate(model_name,task, loader, model, dataset_name, adapt = None, confusion = False):
    model.eval()

    def get_extra_param():
        return None

    perf = PerfTrackVal(task, extra_param=get_extra_param())
    # if dataset_name == "OmniMN40":
    #     subset_list=[0,2,5,6,7,8,10,15,17,18,19,20,26,30,32,33,36,37]
    #     # valid_classes = [0, 2, 5, 6, 7,8, 10, 15, 17, 18,19, 20, 26, 30, 32, 33, 36, 37]
    #     perf = PerfTrackValomni(task,subset_list, extra_param=get_extra_param())

    time_dl = 0
    time_gi = 0
    time_model = 0
    time_upd = 0
    times_model=[]

    with torch.no_grad():
        bar = ProgressBar(max_value=len(loader))
        time5  = time()
        if confusion:
            pred = []
            ground = []
        for i, data_batch in enumerate(loader):
            data_batch={key:data_batch[key].cuda() for key in data_batch}
            time1 = time()
            inp = get_inp(task, model, data_batch, loader.dataset.batch_proc, dataset_name,model_name)
            time2 = time()

            if adapt.METHOD == 'bn':
                model = adapt_bn(inp,model,adapt)
            elif adapt.METHOD == 'tent':
                model = adapt_tent(inp,model,adapt)

            #model(pc=point_cloud, label=labels)model(pc=point_cloud, label=labels)
            out = model(**inp)
            # print("out",out['logit'].shape)#cuda
            if confusion:
                pred.append(out['logit'].squeeze().cpu())
                ground.append(data_batch['label'].squeeze().cpu())

            time3 = time()
            perf.update(data_batch=data_batch, out=out)
            time4 = time()
            ###origin
            time_dl += (time1 - time5)
            time_gi += (time2 - time1)
            time_model += (time3 - time2)
            time_upd += (time4 - time3)

            time5 = time()
            bar.update(i)
            
    ###origin
    print(f"Time DL: {time_dl}, Time Get Inp: {time_gi}, Time Model: {time_model}, Time Update: {time_upd}")
    if not confusion:
        return perf.agg()
    else:
        pred = np.argmax(torch.cat(pred).numpy(), axis=1)
        # print(pred)
        ground = torch.cat(ground).numpy()
        # print(ground)
        return perf.agg(), pred, ground
    
def check_device(obj):
    if isinstance(obj, torch.Tensor):
        return obj.device
    elif isinstance(obj, dict):
        return {key: check_device(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [check_device(item) for item in obj]
    else:
        return None

def train(task, loader, model, optimizer, loss_name, dataset_name, cfg):
    model.train()

    def get_extra_param():
       return None

    perf = PerfTrackTrain(task, extra_param=get_extra_param())
    time_forward = 0
    time_backward = 0
    time_data_loading = 0
    times_forward=[]

    time3  = time()
    for i, data_batch in enumerate(loader):#cpu
        time1 = time()
        data_batch={key:data_batch[key].cuda() for key in data_batch}

        if cfg.AUG.NAME == 'cutmix_r':
            data_batch = aug_utils.cutmix_r(data_batch,cfg)            
        elif cfg.AUG.NAME == 'cutmix_k':
            data_batch = aug_utils.cutmix_k(data_batch,cfg)
        elif cfg.AUG.NAME == 'mixup':
            data_batch = aug_utils.mixup(data_batch,cfg)
        elif cfg.AUG.NAME == 'rsmix':
            data_batch = aug_utils.rsmix(data_batch,cfg)
            data_batch={key:data_batch[key].cuda() for key in data_batch}
        elif cfg.AUG.NAME == 'pgd':
            data_batch = aug_utils.pgd(data_batch,model, task, loss_name, dataset_name)
            
            model.train()
        # print(data_batch)
        inp = get_inp(task, model, data_batch, loader.dataset.batch_proc, dataset_name,cfg.EXP.MODEL_NAME)
        out = model(**inp)
        loss = get_loss(task, loss_name, data_batch, out, dataset_name)

        perf.update_all(data_batch=data_batch, out=out, loss=loss)
        time2 = time()

        if loss.ne(loss).any():
            print("WARNING: avoiding step as nan in the loss")
        else:
            optimizer.zero_grad()
            loss.backward()
            bad_grad = False
            for x in model.parameters():
                if x.grad is not None:
                    if x.grad.ne(x.grad).any():
                        print("WARNING: nan in a gradient")
                        bad_grad = True
                        break
                    if ((x.grad == float('inf')) | (x.grad == float('-inf'))).any():
                        print("WARNING: inf in a gradient")
                        bad_grad = True
                        break

            if bad_grad:
                print("WARNING: avoiding step as bad gradient")
            else:
                optimizer.step()
        ###origin
        time_data_loading += (time1 - time3)
        time_forward += (time2 - time1)
        time3 = time()
        time_backward += (time3 - time2)
        ###origin
        if i % 50 == 0:
            print(
                f"[{i}/{len(loader)}] avg_loss: {perf.agg_loss()}, FW time = {round(time_forward, 2)}, "
                f"BW time = {round(time_backward, 2)}, DL time = {round(time_data_loading, 2)}")
        
    return perf.agg(), perf.agg_loss()


def save_checkpoint(id, epoch, model, optimizer,  lr_sched, bnm_sched, test_perf, cfg):
    model.cpu()
    # path = f"./runs/{cfg.EXP.EXP_ID}/model_{id}.pth"
    path = f"./checkpoints/{cfg.EXP.EXP_ID}/model_{id}.pth"

    torch.save({
        'cfg': vars(cfg),
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'lr_sched_state': lr_sched.state_dict(),
        'bnm_sched_state': bnm_sched.state_dict() if bnm_sched is not None else None,
        'test_perf': test_perf,
    }, path)
    print('Checkpoint saved to %s' % path)
    model.to(DEVICE)


def load_best_checkpoint(model, cfg):
    # path = f"./runs/{cfg.EXP.EXP_ID}/model_best.pth"
    path = f"./checkpoints/{cfg.EXP.EXP_ID}/model_best.pth"

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    print('Checkpoint loaded from %s' % path)

def load_model(model, model_path,model_name):
    print(f'Recovering model and checkpoint from {model_path}')
    checkpoint = torch.load(model_path,map_location=torch.device('cuda:0'))
    # for key in checkpoint:
    #     print(key)
    new_state_dict = {}
    if model_name == "rpc":
        
        # # for k, v in checkpoint['model_state_dict'].items():
        for k, v in checkpoint.items():
            if k.startswith('module.'):  
                new_key = 'module.model.' + k[len('module.'):]  
            else:  
                new_key = 'model.' + k  
            new_state_dict[new_key] = v

            # new_key = 'model.' + k
            # new_state_dict[new_key] = v
    
    elif model_name == "pct":
        for k, v in checkpoint.items():
            new_key = 'module.' + k
            new_state_dict[new_key] = v
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    try:
        if 'model_state_dict' in new_state_dict:
            model.load_state_dict(new_state_dict['model_state_dict'])
        elif 'model_state' in new_state_dict:
            model.load_state_dict(new_state_dict['model_state'])
        else:
            model.load_state_dict(new_state_dict)
    except:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(new_state_dict['model_state'])
        else:
            model = nn.DataParallel(model)
            model.load_state_dict(new_state_dict)

    return model

###origin
# def load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path):
#     print(f'Recovering model and checkpoint from {model_path}')
#     checkpoint = torch.load(model_path)
#     try:
#         model.load_state_dict(checkpoint['model_state'])
#     except:
#         if isinstance(model, nn.DataParallel):
#             model.module.load_state_dict(checkpoint['model_state'])
#         else:
#             print("<<<<<<<<<<<<<")

#             model = nn.DataParallel(model)
#             model.load_state_dict(checkpoint['model_state'])
#             model = model.module

#     optimizer.load_state_dict(checkpoint['optimizer_state'])
#     # for backward compatibility with saved models
#     if 'lr_sched_state' in checkpoint:
#         lr_sched.load_state_dict(checkpoint['lr_sched_state'])
#         if checkpoint['bnm_sched_state'] is not None:
#             bnm_sched.load_state_dict(checkpoint['bnm_sched_state'])
#     else:
#         print("WARNING: lr scheduler and bnm scheduler states are not loaded.")

#     return model

def load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path):
    print(f'Recovering model and checkpoint from {model_path}')
    checkpoint = torch.load(model_path)
    try:
        ###modify
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
    except:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state'])
        else:
            # model = nn.DataParallel(model)
            # model.load_state_dict(checkpoint['model_state'])
            model = nn.DataParallel(model)
            model.load_state_dict(checkpoint)

            # model.load_state_dict(checkpoint['model_state'])
            # model = model.module

    optimizer.load_state_dict(checkpoint['optimizer_state'])
    # for backward compatibility with saved models
    if 'lr_sched_state' in checkpoint:
        lr_sched.load_state_dict(checkpoint['lr_sched_state'])
        if checkpoint['bnm_sched_state'] is not None:
            bnm_sched.load_state_dict(checkpoint['bnm_sched_state'])
    else:
        print("WARNING: lr scheduler and bnm scheduler states are not loaded.")

    return model

def get_model(cfg):
    if cfg.EXP.MODEL_NAME == 'simpleview':
        model = models.MVModel(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            **cfg.MODEL.MV)
    elif cfg.EXP.MODEL_NAME == 'rscnn':
        model = models.RSCNN(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            **cfg.MODEL.RSCNN)
    elif cfg.EXP.MODEL_NAME == 'pointnet2':
        model = models.PointNet2(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            ###modify add
            sample_type=cfg.EXP.sample_type,
            use_upsample=cfg.EXP.use_upsample
            )
            # ,**cfg.MODEL.PN2)
    elif cfg.EXP.MODEL_NAME == 'dgcnn':
        model = models.DGCNN(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            ###modify add
            sample_type=cfg.EXP.sample_type,
            use_upsample=cfg.EXP.use_upsample)
    elif cfg.EXP.MODEL_NAME == 'pointnet':
        model = models.PointNet(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            ###modify add
            sample_type=cfg.EXP.sample_type,
            use_upsample=cfg.EXP.use_upsample)
    elif cfg.EXP.MODEL_NAME == 'pct':
        model = models.Pct(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            ###modify add
            sample_type=cfg.EXP.sample_type,
            use_upsample=cfg.EXP.use_upsample)
    elif cfg.EXP.MODEL_NAME == 'pointMLP':
        model = models.pointMLP(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET)
    elif cfg.EXP.MODEL_NAME == 'pointMLP2':
        model = models.pointMLP2(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET)
    elif cfg.EXP.MODEL_NAME == 'curvenet':
        model = models.CurveNet(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            ###modify add
            sample_type=cfg.EXP.sample_type,
            use_upsample=cfg.EXP.use_upsample)
            
    elif cfg.EXP.MODEL_NAME == 'gdanet':
        model = models.GDANET(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            ###modify add
            sample_type=cfg.EXP.sample_type,
            use_upsample=cfg.EXP.use_upsample)
    elif cfg.EXP.MODEL_NAME == 'rpc':
        model = models.RPC(
            task=cfg.EXP.TASK,
            dataset=cfg.EXP.DATASET,
            ###modify add
            sample_type=cfg.EXP.sample_type,
            use_upsample=cfg.EXP.use_upsample)
    else:
        assert False

    return model


def get_metric_from_perf(task, perf, metric_name):
    if task in ['cls', 'cls_trans']:
        assert metric_name in ['acc']
        metric = perf[metric_name]
    else:
        assert False
    return metric


def get_optimizer(optim_name, tr_arg, model):
    if optim_name == 'vanilla':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tr_arg.learning_rate,
            weight_decay=tr_arg.l2)
        lr_sched = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=tr_arg.lr_decay_factor,
            patience=tr_arg.lr_reduce_patience,
            verbose=True,
            min_lr=tr_arg.lr_clip)
        bnm_sched = None
    elif optim_name == 'pct':
        pass
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tr_arg.learning_rate,
            weight_decay=tr_arg.l2)
        lr_sched = lr_scheduler.CosineAnnealingLR(
            optimizer,
            tr_arg.num_epochs,
            eta_min=tr_arg.learning_rate)
        bnm_sched = None
    else:
        assert False

    return optimizer, lr_sched, bnm_sched


def entry_train(cfg, resume=False, model_path=""):
    loader_train = create_dataloader(split='train', cfg=cfg)
    loader_valid = create_dataloader(split='valid', cfg=cfg)
    loader_test  = create_dataloader(split='test',  cfg=cfg)

    model = get_model(cfg)
    model.to(DEVICE)
    print(model)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    optimizer, lr_sched, bnm_sched = get_optimizer(cfg.EXP.OPTIMIZER, cfg.TRAIN, model)

    if resume:
        model = load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path)
    else:
        assert model_path == ""
    
    # log_dir = f"./runs/{cfg.EXP.EXP_ID}"
    log_dir = f"./checkpoints/{cfg.EXP.EXP_ID}"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tb = TensorboardManager(log_dir)
    track_train = TrackTrain(early_stop_patience=cfg.TRAIN.early_stop)
    # epoch_ = 101
    # for epoch in range(epoch_,cfg.TRAIN.num_epochs):
    for epoch in range(cfg.TRAIN.num_epochs):

        print(f'Epoch {epoch}')

        print('Training..')
        train_perf, train_loss = train(cfg.EXP.TASK, loader_train, model, optimizer, cfg.EXP.LOSS_NAME, cfg.EXP.DATASET, cfg)
        pprint.pprint(train_perf, width=80)
        tb.update('train', epoch, train_perf)

        if (not cfg.EXP_EXTRA.no_val) and epoch % cfg.EXP_EXTRA.val_eval_freq == 0:
                print('\nValidating..')
                val_perf = validate(cfg.EXP.MODEL_NAME,cfg.EXP.TASK, loader_valid, model, cfg.EXP.DATASET, cfg.ADAPT)
                pprint.pprint(val_perf, width=80)
                tb.update('val', epoch, val_perf)
        else:
            val_perf = defaultdict(float)

        if (not cfg.EXP_EXTRA.no_test) and (epoch % cfg.EXP_EXTRA.test_eval_freq == 0):
            print('\nTesting..')
            test_perf = validate(cfg.EXP.MODEL_NAME,cfg.EXP.TASK, loader_test, model, cfg.EXP.DATASET, cfg.ADAPT)
            pprint.pprint(test_perf, width=80)
            tb.update('test', epoch, test_perf)
        else:
            test_perf = defaultdict(float)

        track_train.record_epoch(
            epoch_id=epoch,
            train_metric=get_metric_from_perf(cfg.EXP.TASK, train_perf, cfg.EXP.METRIC),
            val_metric=get_metric_from_perf(cfg.EXP.TASK, val_perf, cfg.EXP.METRIC),
            test_metric=get_metric_from_perf(cfg.EXP.TASK, test_perf, cfg.EXP.METRIC))

        if (not cfg.EXP_EXTRA.no_val) and track_train.save_model(epoch_id=epoch, split='val'):
            print('Saving best model on the validation set')
            save_checkpoint('best_val', epoch, model, optimizer,  lr_sched, bnm_sched, test_perf, cfg)

        if (not cfg.EXP_EXTRA.no_test) and track_train.save_model(epoch_id=epoch, split='test'):
            print('Saving best model on the test set')
            save_checkpoint('best_test', epoch, model, optimizer,  lr_sched, bnm_sched, test_perf, cfg)

        if (not cfg.EXP_EXTRA.no_val) and track_train.early_stop(epoch_id=epoch):
            print(f"Early stopping at {epoch} as val acc did not improve for {cfg.TRAIN.early_stop} epochs.")
            break

        if (not (cfg.EXP_EXTRA.save_ckp == 0)) and (epoch % cfg.EXP_EXTRA.save_ckp == 0):
            save_checkpoint(f'{epoch}', epoch, model, optimizer,  lr_sched, bnm_sched, test_perf, cfg)

        if cfg.EXP.OPTIMIZER == 'vanilla':
            assert bnm_sched is None
            lr_sched.step(train_loss)
        else:
            lr_sched.step()

    print('Saving the final model')
    save_checkpoint('final', epoch, model, optimizer,  lr_sched, bnm_sched, test_perf, cfg)

    print('\nTesting on the final model..')
    last_test_perf = validate(cfg.EXP.MODEL_NAME,cfg.EXP.TASK, loader_test, model, cfg.EXP.DATASET, cfg.ADAPT)
    pprint.pprint(last_test_perf, width=80)

    tb.close()

###origin
# def entry_test(cfg, test_or_valid, model_path="", confusion = False):
###modify
def entry_test(cfg, test_or_valid, model_path="", confusion = False, addpref = False):

    split = "test" if test_or_valid else "valid"
    loader_test = create_dataloader(split=split, cfg=cfg)

    model = get_model(cfg)
    model.to(DEVICE)
    print(model)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    optimizer, lr_sched, bnm_sched = get_optimizer(cfg.EXP.OPTIMIZER, cfg.TRAIN, model)
    ###origin
    # model = load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path)
    ###modify
    if addpref:
        model = load_model(model, model_path,cfg.EXP.MODEL_NAME)
    else:
        model = load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path)

    model.eval()
    if confusion:
        test_perf, pred, ground = validate(cfg.EXP.MODEL_NAME,cfg.EXP.use_sample,cfg.EXP.TASK, loader_test, model, cfg.EXP.DATASET, cfg.ADAPT, confusion)
        print(pred.shape, ground.shape)
        #### some hardcoding #######
        np.save('./output/' + cfg.EXP.MODEL_NAME + '_' +  cfg.DATALOADER.MODELNET40_C.corruption + '_' + str(cfg.DATALOADER.MODELNET40_C.severity)  + '_pred.npy',pred )
        np.save('./output/' + cfg.EXP.MODEL_NAME + '_' +  cfg.DATALOADER.MODELNET40_C.corruption + '_' + str(cfg.DATALOADER.MODELNET40_C.severity)  + '_ground.npy',ground)
        #### #### #### #### #### ####
    else:
        test_perf = validate(cfg.EXP.MODEL_NAME,cfg.EXP.TASK, loader_test, model, cfg.EXP.DATASET, cfg.ADAPT, confusion)
    #输入到文件里
    ###origin
    # print("Model: {} Corruption: {} Severity: {} Acc: {} Class Acc: {}".format(cfg.EXP.MODEL_NAME, cfg.DATALOADER.MODELNET40_C.corruption, cfg.DATALOADER.MODELNET40_C.severity,test_perf['acc'],test_perf['class_acc']),file=file_object,flush=True)
    ###modify
    if cfg.EXP.DATASET in ["modelnet40_c","pointcloud_c"]:
        print("Model: {} use_upsample: {} sample_type: {} Corruption: {} Severity: {} Acc: {} Class Acc: {}"
              .format(cfg.EXP.MODEL_NAME,cfg.EXP.use_upsample,cfg.EXP.sample_type, cfg.DATALOADER.MODELNET40_C.corruption, cfg.DATALOADER.MODELNET40_C.severity,test_perf['acc'],test_perf['class_acc']),file=file_object,flush=True)
    else:
        print("Model: {} DataSet: {} use_upsample: {} sample_type: {} Acc: {} Class Acc: {}"
              .format(cfg.EXP.MODEL_NAME,cfg.EXP.DATASET,cfg.EXP.use_upsample,cfg.EXP.sample_type,test_perf['acc'],test_perf['class_acc']),file=file_object,flush=True)

    pprint.pprint(test_perf, width=80)
    return test_perf


def rscnn_vote_evaluation(cfg, model_path, log_file):
    model = get_model(cfg)
    checkpoint = torch.load(model_path)
    try:
        model.load_state_dict(checkpoint['model_state'])
    except:
        print("WARNING: using dataparallel to load data")
        model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['model_state'])
    print(f"Checkpoint loaded from {model_path}")
    model.to(DEVICE)
    model.eval()

    assert cfg.EXP.DATASET in ["modelnet40_rscnn"]
    loader_test = create_dataloader(split='test', cfg=cfg)

    rscnn_voting_evaluate_cls(
        loader=loader_test,
        model=model,
        data_batch_to_points_target=lambda x: (x['pc'], x['label']),
        points_to_inp=lambda x: {'pc': x},
        out_to_prob=lambda x: F.softmax(x['logit'], dim=1),
        log_file=log_file
    )

def pn2_vote_evaluation(cfg, model_path, log_file):
    assert cfg.EXP.DATASET in ["modelnet40_pn2"]
    loader_test = create_dataloader(split='test', cfg=cfg)

    model = get_model(cfg)
    checkpoint = torch.load(model_path)
    try:
        model.load_state_dict(checkpoint['model_state'])
    except:
        print("WARNING: using dataparallel to load data")
        model = nn.DataParallel(model)
        model.load_state_dict(checkpoint['model_state'])
    print(f"Checkpoint loaded from {model_path}")
    model.to(DEVICE)
    model.eval()

    pn2_vote_evaluate_cls(loader_test, model, log_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())
    parser.add_argument('--entry', type=str, default="train")
    parser.add_argument('--exp-config', type=str, default="")
    parser.add_argument('--model-path', type=str, default="")
    parser.add_argument('--resume', action="store_true", default=False)
    parser.add_argument('--gpu',type=str,default='3',help="Which gpu to use")
    parser.add_argument('--corruption',type=str,default='uniform',
                        help="Which corruption to use")
    parser.add_argument('--output',type=str,default='./test.txt',
                        help="path to output file")
    parser.add_argument('--severity',type=int,default=1,
                        help="Which severity to use")

    parser.add_argument('--confusion', action="store_true", default=False,
                        help="whether to output confusion matrix data")
    ###modify add
    parser.add_argument('--add_prefix', action="store_true", default=False,
                        help="whether to add prefix to load model")
    parser.add_argument('--use_upsample', type=str, default='no', metavar='N',
                        choices=['no','lgp_or_lgd','clgp'])### lgp_or_lgd for train,clgp for test
    parser.add_argument('--sample_type', type=str, default='fps', metavar='N',
                        choices=['fps','wrs','ffps'])### wrs for train, ffps for test

    cmd_args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = cmd_args.gpu

    if cmd_args.entry == "train":
        assert not cmd_args.exp_config == ""
        if not cmd_args.resume:
            assert cmd_args.model_path == ""

        cfg = get_cfg_defaults()

        cfg.merge_from_file(cmd_args.exp_config)
        ###modify add
        cfg.EXP.sample_type = cmd_args.sample_type
        if cfg.EXP.MODEL_NAME in ['pct','pointnet2','pointnet','dgcnn','gdanet','curvenet','rpc']:
            cfg.EXP.use_upsample = cmd_args.use_upsample
        else:
            cfg.EXP.use_upsample = 'no'

        # if cfg.EXP.EXP_ID == "":
            # cfg.EXP.EXP_ID = str(datetime.now())[:-7].replace(' ', '-')
        cfg.EXP.EXP_ID = f"{cfg.EXP.EXP_ID}_{cmd_args.use_upsample}_{cmd_args.sample_type}"
        cfg.freeze()
        print(cfg)

        random.seed(cfg.EXP.SEED)
        np.random.seed(cfg.EXP.SEED)
        torch.manual_seed(cfg.EXP.SEED)

        entry_train(cfg, cmd_args.resume, cmd_args.model_path)
    elif cmd_args.entry in ["test", "valid","oo3","omni"]:
        # print("path:",cmd_args.model_path)
        ###origin
        # file_object = open(cmd_args.output, 'a')
        test_or_valid = cmd_args.entry == "test"
        ###modify
        output_path = '/'.join(cmd_args.model_path.split('/')[:-1])
        ###add for different epoch model
        # file_object = open(f'{output_path}/test_{cmd_args.use_upsample}_{cmd_args.sample_type}.txt', 'a')
        mname = (lambda m: m.group(1) if m else '')(re.search(r'model_(.*?)\.pth', cmd_args.model_path))
        

        assert not cmd_args.exp_config == ""
        assert not cmd_args.model_path == ""

        cfg = get_cfg_defaults()         
        cfg.merge_from_file(cmd_args.exp_config)
        ###modify and add
        
        cfg.EXP.sample_type = cmd_args.sample_type
        if cfg.EXP.MODEL_NAME in ['pct','pointnet2','pointnet','dgcnn','gdanet','curvenet','rpc']:
            cfg.EXP.use_upsample = cmd_args.use_upsample
        else:
            cfg.EXP.use_upsample = 'no'
        
        ###add for OmniMN40
        if cmd_args.entry in ["oo3"]:
            test_or_valid = "test"
            cfg.EXP.DATASET = "OmniMN40"
            cfg.DATALOADER.MODELNET40_C.test_data_path = "/home/user_tp/workspace/code/attack/ModelNet40-C/data/OmniMN40/OmniMN40.h5"
            file_object = open(f'{output_path}/oo3_{cmd_args.use_upsample}_{cmd_args.sample_type}_{mname}.txt', 'a')
        elif cmd_args.entry in ["omni"]:
            test_or_valid = "test"
            cfg.EXP.DATASET = "OmniMN40"
            cfg.DATALOADER.MODELNET40_C.test_data_path = "/home/user_tp/workspace/data/OmniMN40desk/OmniMN40.h5"
            file_object = open(f'{output_path}/omni_{cmd_args.use_upsample}_{cmd_args.sample_type}_{mname}.txt', 'a')
        else:
            file_object = open(f'{output_path}/test_{cmd_args.use_upsample}_{cmd_args.sample_type}_{mname}.txt', 'a')

        cfg.freeze()
        print(cfg)

        random.seed(cfg.EXP.SEED)
        np.random.seed(cfg.EXP.SEED)
        torch.manual_seed(cfg.EXP.SEED)

        
        ###origin
        # entry_test(cfg, test_or_valid, cmd_args.model_path,cmd_args.confusion)
        ###modify
        entry_test(cfg, test_or_valid, cmd_args.model_path,cmd_args.confusion,cmd_args.add_prefix)
    ##add elif
    elif cmd_args.entry in ["mnc","pcc","oo3dc","omnic"]:
        # print("path:",cmd_args.model_path)
        ###add for different epoch model
        mname = (lambda m: m.group(1) if m else '')(re.search(r'model_(.*?)\.pth', cmd_args.model_path))
        output_path = '/'.join(cmd_args.model_path.split('/')[:-1])

        assert not cmd_args.exp_config == ""
        assert not cmd_args.model_path == ""

        cfg = get_cfg_defaults()

        cfg.merge_from_file(cmd_args.exp_config)
        cfg.DATALOADER.MODELNET40_C.corruption = cmd_args.corruption
        cfg.DATALOADER.MODELNET40_C.severity = cmd_args.severity
        if cmd_args.entry == "mnc":
            file_object = open(f'{output_path}/mnc_{cmd_args.use_upsample}_{cmd_args.sample_type}_{mname}.txt', 'a')
            data_path = "./data/modelnet40_c/"
        elif cmd_args.entry == "pcc":
            cfg.EXP.DATASET = "pointcloud_c"
            file_object = open(f'{output_path}/pcc_{cmd_args.use_upsample}_{cmd_args.sample_type}_{mname}.txt', 'a')
            data_path = "./data/modelnet_c/"
            # cfg.DATALOADER.MODELNET40_C.corruption = 'scale'
            # cfg.DATALOADER.MODELNET40_C.severity = 0
        elif cmd_args.entry == "omnic":
            cfg.EXP.DATASET = "pointcloud_c"
            file_object = open(f'{output_path}/omnic_{cmd_args.use_upsample}_{cmd_args.sample_type}_{mname}.txt', 'a')
            data_path = "./data/omni_c/"

        print("corruption",cfg.DATALOADER.MODELNET40_C.corruption)

        ###modify add
        cfg.DATALOADER.MODELNET40_C.test_data_path = data_path
        cfg.EXP.sample_type = cmd_args.sample_type
        if cfg.EXP.MODEL_NAME in ['pct','pointnet2','pointnet','dgcnn','gdanet','curvenet','rpc']:
            cfg.EXP.use_upsample = cmd_args.use_upsample
        else:
            cfg.EXP.use_upsample = 'no'
        cfg.freeze()
        print(cfg)
        random.seed(cfg.EXP.SEED)
        np.random.seed(cfg.EXP.SEED)
        torch.manual_seed(cfg.EXP.SEED)

        entry_test(cfg, True, cmd_args.model_path,cmd_args.confusion,cmd_args.add_prefix)
         
        if torch.cuda.is_available():  
            torch.cuda.empty_cache()  # 尝试清空CUDA缓存
    ###end

    elif cmd_args.entry in ["rscnn_vote", "pn2_vote"]:
        assert not cmd_args.exp_config == ""
        assert not cmd_args.model_path == ""
        log_file = f"vote_log/{cmd_args.model_path.replace('/', '_')}_{cmd_args.entry.replace('/', '_')}.log"

        cfg = get_cfg_defaults()
        cfg.merge_from_file(cmd_args.exp_config)
        cfg.freeze()
        print(cfg)

        seed  = cfg.EXP.SEED
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        if cmd_args.entry == "rscnn_vote":
            rscnn_vote_evaluation(cfg, cmd_args.model_path, log_file)
        elif cmd_args.entry == "pn2_vote":
            pn2_vote_evaluation(cfg, cmd_args.model_path, log_file)
    else:
        assert False
