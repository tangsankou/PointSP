"""
@Author: Pin Tang
@Contact: tangpin1874@163.com
@File: sampling.py
@Time: 2025/5/6 21:39 PM
"""
import torch
import os
import numpy as np
import pandas as pd
from .interpolation import Interpolation
import open3d as o3d

def upsample_point_cloud(point_cloud, ratio, step_size,distance, idx_k, normals):
    """ Upsample the point cloud by adding noise and randomly selecting points. """
    # Simulate upsampling with noise and random selection
    num_points = point_cloud.shape[1]
    I = Interpolation(step_size)
    pc = I.random_k_neighbors_shape_invariant_perturb(point_cloud,distance, idx_k, normals)
    # Select random indices from the noise array
    num_points_to_select = int((ratio - 1) * num_points)
    sampled_indices = torch.randperm(num_points)[:num_points_to_select]

    # Concatenate selected noisy points with the original point cloud
    upsampled_cloud = torch.cat([point_cloud, pc[:, sampled_indices]], dim=1)

    return upsampled_cloud#,pc[:, sampled_indices]

###24-5-21 
###改进的drop方法，global_score决定drop点的(0.0-1.0)local-global范围
###origin
#def downsample_point_cloud_score(point_cloud,r):
def downsample_point_cloud_score(point_cloud,r):#,global_score):
    B, N, _ = point_cloud.shape
    k = int(N * (1 - r))
    # global_score=1
    global_score = torch.rand(1)#origin
    center_indices = torch.randint(0, N, (B,))

    center_points = point_cloud[torch.arange(B), center_indices]
    distances = torch.norm(point_cloud - center_points[:, None, :], dim=2)
    sorted_indices = distances.argsort(dim=1)

    new_k = int((N - k) * global_score) + k
    top_k_indices = sorted_indices[:, torch.randperm(new_k)[:k]]

    mask = torch.ones(B, N, dtype=torch.bool)
    batch_indices = torch.arange(B).unsqueeze(1).expand(B, k)
    mask[batch_indices, top_k_indices] = False
    filtered_point_cloud = point_cloud[mask].view(B, N - k, 3)
    return filtered_point_cloud


###用了两种改变
def process_point_cloud_mix(point_cloud, step_size, normals):
    ratio = 2 ** ((torch.rand(1) - 0.5) * 2)  # 将随机数映射到 [0.5, 2] 范围内      
    if ratio <= 1:
        processed_cloud = downsample_point_cloud_score(point_cloud, ratio)
    else:
        # Upsample the point cloud
        distance, idx_k = knn(point_cloud, k=20)
        processed_cloud = upsample_point_cloud(point_cloud, ratio, step_size,distance, idx_k, normals)
    return processed_cloud

def get_normal_vector(points):
    """Calculate the normal vector.
    Args:
        points (torch.cuda.FloatTensor): the point cloud with N points, [N, 3].
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.squeeze(0).detach().cpu().numpy())
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    normal_vec = torch.FloatTensor(np.asarray(pcd.normals)).cuda().unsqueeze(0)
    return normal_vec

#@SageMix/pointcloud/model.py
def knn(x, k):
    """
    Input:
        x: pointcloud data, [B, N, C]
        k: number of knn
    Return:
        pairwise_distance: distance of points, [B, N, N]
        idx: index of points' knn, [B, N, k]
    """
    x = x.transpose(2, 1) #[B, N, C]->[B, C, N]
    inner = -2*torch.matmul(x.transpose(2, 1), x) #(B, N, N)
    xx = torch.sum(x**2, dim=1, keepdim=True) #(B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1) #(B, N, N)
    # print("pairwise_distance:",pairwise_distance.shape)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (B, N, k)
    return -1*pairwise_distance, idx


#@Pointnet_Pointnet2_pytorch/models/pointnet2_utils.py
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

#@Pointnet_Pointnet2_pytorch/models/pointnet2_utils.py
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3](质心)
    Return:
        group_idx: grouped points index, [B, S, nsample]
        pairwise_distance: distance of points, [B, N, N]
        idx: index of points' knn, [B, N, k]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    # print("sqrdists",sqrdists.shape)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    # print("group_idx",group_idx.shape)
    return sqrdists, group_idx

def cal_weight(xyz, k=16):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        k: k of knn or max sample number in local region of ball_query
    Return:
        weights: weights of pointcloud, [B, N]
        idx: index of points' knn, [B, N, k]
    """
    B, N, C = xyz.shape

    pairwise_distance, idx = knn(xyz, k)
    #索引出k近邻的距离    
    distance = torch.gather(pairwise_distance, dim=-1, index=idx)#(B, N, k)
    # 沿着维度k计算平均值
    mean_along_k = torch.mean(distance, dim=-1)
    threshold = torch.quantile(mean_along_k, q=0.5, dim=-1)
    weights = torch.zeros_like(distance, dtype=torch.float32)  # shape: [B, N, k]
    weights[distance <= threshold.view(B, 1, 1)] = 1. #(B,N,k)
    weights = torch.sum(weights, dim=-1,keepdim=False) #(B, N)
    weights /= torch.sum(weights, dim=-1).view(B, 1)#/1024 # 防止除以零, (B, N) 
    return weights,idx

###visual
def weight_max(xyz, k=20):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        k: k of knn or max sample number in local region of ball_query
    Return:
        weights: weights of pointcloud, [B, N]
        idx: index of points' knn, [B, N, k]
    """
    B, N, C = xyz.shape
    pairwise_distance, idx = knn(xyz, k)
    
    distance = torch.gather(pairwise_distance, dim=-1, index=idx)#(B,N,N)->(B, N, k)
    mean_along_k = torch.max(distance, dim=-1)[0]
    threshold = torch.quantile(mean_along_k, q=0.5, dim=-1)
    weights = torch.zeros_like(distance, dtype=torch.float32)  # shape: [B, N, k]
    weights[distance <= threshold.view(B, 1, 1)] = 1. #(B,N,k)
    isolated_rate=torch.zeros_like(mean_along_k,dtype=torch.float32)
    isolated_rate[mean_along_k<=threshold.view(B,1)]=1#[B,N]
    new_weights=torch.zeros_like(mean_along_k,dtype=torch.float32)
    new_weights[mean_along_k<=threshold.view(B,1)]=1#[B,N]
    new_weights=new_weights.view(B, N, 1).repeat(1, 1, N).transpose(1,2)
    is_weights = torch.gather(new_weights, dim=-1, index=idx)  
    is_weights=torch.sum(is_weights,dim=-1,keepdim=False)

    weights = torch.sum(weights, dim=-1,keepdim=False) #(B, N)
    
    return weights,isolated_rate,is_weights
def weight(xyz, k=20):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        k: k of knn or max sample number in local region of ball_query
    Return:
        weights: weights of pointcloud, [B, N]
        idx: index of points' knn, [B, N, k]
    """
    B, N, C = xyz.shape
    # 计算每个点对之间的距离值及knn索引
    pairwise_distance, idx = knn(xyz, k)
    #索引出k近邻的距离    
    distance = torch.gather(pairwise_distance, dim=-1, index=idx)#(B,N,N)->(B, N, k)
    # 沿着维度k计算平均值
    mean_along_k = torch.mean(distance, dim=-1)
    threshold = torch.quantile(mean_along_k, q=0.5, dim=-1)
    weights = torch.zeros_like(distance, dtype=torch.float32)
    weights[distance <= threshold.view(B, 1, 1)] = 1. #(B,N,k)
    isolated_rate=torch.zeros_like(mean_along_k,dtype=torch.float32)
    isolated_rate[mean_along_k<=threshold.view(B,1)]=1#[B,N]
    new_weights=torch.zeros_like(mean_along_k,dtype=torch.float32)
    new_weights[mean_along_k<=threshold.view(B,1)]=1#[B,N]
    new_weights=new_weights.view(B, N, 1).repeat(1, 1, N).transpose(1,2)
    is_weights = torch.gather(new_weights, dim=-1, index=idx)  
    is_weights=torch.sum(is_weights,dim=-1,keepdim=False)

    weights = torch.sum(weights, dim=-1,keepdim=False) #(B, N)
    
    return weights,isolated_rate,is_weights


def seva_weights(weights, k, save_path):
    # 假设 k_values 和 weights 是你的字典
    data_dict = {'k': k, 'weights': weights}
    # 将字典转换为 DataFrame
    df = pd.DataFrame(data_dict)
    path = save_path + "k-weights.csv"
    # 保存 DataFrame 到 CSV 文件
    df.to_csv(path, index=False)

def weighted_random_point_sample(xyz, npoint, k=20,replace=False):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
        k: k of knn
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # print("k::::",k)
    device = xyz.device
    B, N, C = xyz.shape	
    centroids = torch.zeros(B, npoint, dtype=torch.long)#.to(device)
    # 计算采样权重
    import time
   
    weights, idx = cal_weight(xyz, k)  # 假设这个函数已经定义且返回(B, N)的weights和idx
       
    centroids = torch.multinomial(weights, npoint, replacement=replace).squeeze()  
    return centroids,idx


if __name__ == '__main__':
    # 测试代码
    # comwesk()
    b = 4
    n = 1024
    point_cloud = torch.rand(b, 20, 3)  # Generate a random point cloud (batch size 1)
    print("pointcloud:",point_cloud)
    normals = torch.zeros_like(point_cloud)
    step_size = 0.03
    k=10
    r=0.8
    idx_k= torch.randint(0, 20, (b, 20, 5))
   
    