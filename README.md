# Enhancing Sampling Protocol for Point Cloud Classification Against Corruptions
This repo contains Pytorch implementation for the paper [Enhancing Sampling Protocol for Point Cloud Classification Against Corruptions](https://arxiv.org/abs/2408.12062)(IJCAI 2025). This codebase is based on [ModelNet40-C](https://github.com/jiachens/ModelNet40-C), and we thank the authors for their great contributions.

## PointSP
![image](https://github.com/tangsankou/PointSP/blob/main/img/main.jpg)
## Overview
#### Models: 
PointNet, PointNet++, PCT, GDANet, CurveNet
#### Training Dataset: 
ModelNet40
#### Test Datasets:
- ModelNet40-C (https://github.com/jiachens/ModelNet40-C): introduces 15 corruption types (occlusion, noise, etc.), each with 5 severity levels, applied to ModelNet40. 
- PointCloud-C (https://github.com/ldkong1205/PointCloud-C): features 7 real-world inspired corruptions (each with 5 levels) on the same data. 
- OmniObject-C (https://github.com/omniobject3d/OmniObject3D): applies PointCloud-C's 7 corruption types (5 levels each) to OmniObject3D classes matching ModelNet40 categories.

## Getting Started

First clone the repository. We would refer to the directory containing the code as `PointSP`.

```
git clone --recurse-submodules git@github.com:tangsankou/PointSP.git
```

## Requirements
```h5py==2.10.0
progressbar2==3.20.0
tensorboardX==2.0
torch==2.0.1 
torchvision==0.15.2 
torchaudio==2.0.2
open3d
Ninja
python==3.8
CUDA >=11.7, 
CuDNN 
GCC
```
We recommend using these versions especially for installing [pointnet++ custom CUDA modules](https://github.com/erikwijmans/Pointnet2_PyTorch/tree/22e8cf527b696b63b66f3873d80ae5f93744bdef).

## Install
```
conda create --name pointsp python=3.8
conda activate pointsp
pip install -r requirements.txt
```

For PointNet++, we need to install custom CUDA modules. Make sure you have access to a GPU during this step. You might need to set the appropriate `TORCH_CUDA_ARCH_LIST` environment variable depending on your GPU model. The following command should work for most cases `export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"`. However, if the install fails, check if `TORCH_CUDA_ARCH_LIST` is correctly set. 

Third-party modules `pointnet2_pyt`, `PCT_Pytorch`, `emd`, and `PyGeM` can be installed by the following script.

```
./setup.sh
```
 
## Usage
This repository provides code for training and evaluating point cloud models (PointNet, PointNet++, PCT, GDANet, CurveNet) on ModelNet40 and testing their robustness against corruptions in ModelNet40-C, PointCloud-C, and  OmniObject-C datasets.To train or test any model, we use the `main.py` script. The format for running this script is as follows. 
## Training
Train a model with upsampling and weighted sampling:
```
CUDA_VISIBLE_DEVICES=0 python main.py \
  --exp-config configs/<cfg_name>.yaml \
  --use_upsample lgp_or_lgd \
  --sample_type wrs  
```
- The --use_upsample parameter is designed for the full-point sampling module in PointSP, while the --sample_type parameter controls the key-point sampling module in PointSP.
- <cfg_name>: Config file (e.g., dgcnn_pointnet2_run_1.yaml).
- lgp_or_lgb: Choose lgp (Local Geometric Propagation)_or_lgd (Local Global Downsampling).
- wrs: Weighted Random Sampling.
## Testing on Corrupted Datasets
```
CUDA_VISIBLE_DEVICES=0 python main.py \
  --entry mnc \(pcc\ominc)
  --model-path <cor_exp/runs>/<cfg_name>/<model_name>.pth \
  --exp-config configs/<cfg_name>.yaml \
  --use_upsample clgp \
  --sample_type ffps \
  --severity 1 \
  --corruption occlusion  
```
- clgp: Conditional Local Geometric Preserved.
- ffps: Filtered Furthest Point Sampling.
- severity: Corruption level (1–5 for ModelNet40-C and 0-4 for PointCloud-C、 OmniObject-C).
- corruption: Type (occlusion etc.).

## Citation
Please cite our paper and ModelNet40-C if you find our work useful in your research. Thank you!
```
@article{Li2025enhancing,
      title={Enhancing Sampling Protocol for Point Cloud Classification Against Corruptions}, 
      author={Chongshou Li, Pin Tang, Xinke Li, Yuheng Liu, Tianrui Li},
      journal={https://arxiv.org/abs/2408.12062},
      year={2025}
}
```
