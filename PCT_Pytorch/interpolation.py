import open3d as o3d
import torch
import numpy as np
# from data import ModelNet40
from torch import nn

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

class Interpolation(nn.Module):
    def __init__(self, step_size):
        super(Interpolation, self).__init__()
        self.step_size = step_size
        
    def get_normal_vector(self, points):
        """Calculate the normal vector.
        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [N, 3].
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.squeeze(0).detach().cpu().numpy())
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        normal_vec = torch.FloatTensor(np.asarray(pcd.normals)).cuda().unsqueeze(0)
        return normal_vec
    
    def get_spin_axis_matrix(self, normal_vec):
        """Calculate the spin-axis matrix.
        Args:
            normal_vec (torch.cuda.FloatTensor): the normal vectors for all N points, [1, N, 3].
        """
        _, N, _ = normal_vec.shape
        x = normal_vec[:,:,0] # [1, N]
        y = normal_vec[:,:,1] # [1, N]
        z = normal_vec[:,:,2] # [1, N]
        assert abs(normal_vec).max() <= 1
        u = torch.zeros(1, N, 3, 3).cuda()
        denominator = torch.sqrt(1-z**2) # \sqrt{1-z^2}, [1, N]
        u[:,:,0,0] = y / denominator
        u[:,:,0,1] = - x / denominator
        u[:,:,0,2] = 0.
        u[:,:,1,0] = x * z / denominator
        u[:,:,1,1] = y * z / denominator
        u[:,:,1,2] = - denominator
        u[:,:,2] = normal_vec
        # revision for |z| = 1, boundary case.
        pos = torch.where(abs(z ** 2 - 1) < 1e-4)[1]
        u[:,pos,0,0] = 1 / np.sqrt(2)
        u[:,pos,0,1] = - 1 / np.sqrt(2)
        u[:,pos,0,2] = 0.
        u[:,pos,1,0] = z[:,pos] / np.sqrt(2)
        u[:,pos,1,1] = z[:,pos] / np.sqrt(2)
        u[:,pos,1,2] = 0.
        u[:,pos,2,0] = 0.
        u[:,pos,2,1] = 0.
        u[:,pos,2,2] = z[:,pos]
        return u.data

    def get_transformed_point_cloud(self, points, normal_vec):
        """Calculate the spin-axis matrix.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 3].
            normal_vec (torch.cuda.FloatTensor): the normal vectors for all N points, [1, N, 3].
        """
        intercept = torch.mul(points, normal_vec).sum(-1, keepdim=True) # P \cdot N, [1, N, 1]
        spin_axis_matrix = self.get_spin_axis_matrix(normal_vec) # U, [1, N, 3, 3]
        translation_matrix = torch.mul(intercept, normal_vec).data # (P \cdot N) N, [1, N, 3]
        new_points = points + translation_matrix #  P + (P \cdot N) N, [1, N, 3]
        new_points = new_points.unsqueeze(-1) # P + (P \cdot N) N, [1, N, 3, 1]
        new_points = torch.matmul(spin_axis_matrix, new_points) # P' = U (P + (P \cdot N) N), [1, N, 3, 1]
        new_points = new_points.squeeze(-1).data # P', [1, N, 3]
        return new_points, spin_axis_matrix, translation_matrix

    def get_original_point_cloud(self, new_points, spin_axis_matrix, translation_matrix):
        """Calculate the spin-axis matrix.

        Args:
            new_points (torch.cuda.FloatTensor): the transformed point cloud with N points, [1, N, 3].
            spin_axis_matrix (torch.cuda.FloatTensor): the rotate matrix for transformation, [1, N, 3, 3].
            translation_matrix (torch.cuda.FloatTensor): the offset matrix for transformation, [1, N, 3, 3].
        """
        inputs = torch.matmul(spin_axis_matrix.transpose(-1, -2), new_points.unsqueeze(-1)) # U^T P', [1, N, 3, 1]
        inputs = inputs - translation_matrix.unsqueeze(-1) # P = U^T P' - (P \cdot N) N, [1, N, 3, 1]
        inputs = inputs.squeeze(-1) # P, [1, N, 3]
        return inputs

    def shape_invariant_perturb(self,points, normal_vec):
        """Black-box I-FGSM based on shape-invariant sensitivity maps.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        normal_vec = normal_vec / torch.sqrt(torch.sum(normal_vec ** 2, dim=-1, keepdim=True)) # N, [1, N, 3]
        points = points[:,:,:3].data # P, [1, N, 3]
        ori_points = points.data
        #clip_func = ClipPointsLinf(budget=self.eps)# * np.sqrt(3*1024))

        if True: 
            # P -> P', detach()
            new_points, spin_axis_matrix, translation_matrix = self.get_transformed_point_cloud(points, normal_vec)
            new_points = new_points.detach()
            new_points.requires_grad = True
            grad = torch.randn(new_points.shape).to(new_points.device)
            grad[:,:,2] = 0.
            norm = torch.sum(grad ** 2, dim=[1, 2]) ** 0.5
            new_points = new_points - self.step_size * np.sqrt(3*1024) * grad / (norm[:, None, None] + 1e-9)
            points = self.get_original_point_cloud(new_points, spin_axis_matrix, translation_matrix) # P, [1, N, 3]
            #points = clip_func(points, ori_points)
            # points = torch.min(torch.max(points, ori_points - self.eps), ori_points + self.eps) # P, [1, N, 3]

        with torch.no_grad():
            adv_points = points.data
        return adv_points

    def random_k_neighbors_shape_invariant_perturb(self, x,distance, idx_k, normals):
    # def random_k_neighbors_shape_invariant_perturb(self,x_old ,x, idx_k, normals):
        B, N, C = x.shape
        
        # distance,idx_k = knn(x, 20)  # B*N*k
        distance = torch.gather(distance, dim=-1, index=idx_k)#(B, N, k)
        # distance = torch.clamp(distance,min=0.0)
        mean_along_k = torch.sqrt(torch.median(distance, dim=-1,keepdim=True).values)
        # print("mean_along_k:",mean_along_k.shape)
        # 沿着维度k计算平均值
        # mean_along_k = torch.median(distance, dim=-1,keepdim=True).values
        # print("mean:",mean_along_k.shape)
        # 为每个点随机选择一个邻近点
        rand_k_indices = torch.randint(0, idx_k.shape[-1], (B, N, 1), device=x.device)
        gather_indices = torch.gather(idx_k, 2, rand_k_indices)  # B*N*1
        # 扩展gather_indices到与x相同的形状，以便可以使用torch.gather
        gather_indices_expanded = gather_indices.expand(-1, -1, C)

        # 重新排列x以便于按gather_indices_expanded的索引来gather
        ###modify
        x_rearranged = x.gather(1, gather_indices_expanded)#随机临近点
        # print("x_rearranged:",x_rearranged)
        x_vec = x_rearranged - x #[B,N,3] 0.00

        # # 处理x_vec 中接近 0 的元素
        # x_vec_max = torch.max(x_vec)
        # x_vec_min = torch.min(x_vec)
        # # 将 x_vec 中接近 0 的元素替换为在 x_vec 的最大值和最小值之间的随机数
        # zero_mask = torch.abs(x_vec) < 1e-6
        # random_values = torch.rand_like(x_vec) * (x_vec_max - x_vec_min) + x_vec_min
        # x_vec = torch.where(zero_mask, random_values, x_vec)

        # print("x_vec:",x_vec)
        normal_x_vec_proj = torch.sum(x_vec * normals,dim=-1, keepdim=True) * normals
        x_perturb = x_vec - normal_x_vec_proj
        # print("x_perturb:",x_perturb)
        # 计算距离    #distances = torch.sqrt(((x - x_rearranged) ** 2).sum(-1))
        norms = torch.norm(x_perturb, dim=-1, keepdim=True)
        norms = torch.clamp(norms,min=1e-6)
        # print("norm:",norms.shape)
        # halved_norms = norms / 2
        halved_norms = norms / 2
        mask = halved_norms>mean_along_k
        halved_norms[mask] = mean_along_k[mask]
        clamped_norms = halved_norms
        # clamped_norms = torch.clamp(halved_norms, max=self.step_size)
        # print("norms:",norms)# 0.00
        # print("clamped_norms:",clamped_norms.shape) #0.00
        perturb = x_perturb/ norms * clamped_norms
        # perb_x = torch.mul(x, perturb)
        # print("perturb:",perturb)#nan
        perb_x = x + perturb
        return perb_x
    
if __name__ == '__main__':
    
    from sampling import weighted_random_point_sample
    I = Interpolation(step_size=0.02)
    # pts = np.random.rand(1024,3).astype(np.float32)
    # normals = I.get_normal_vector(torch.from_numpy(pts).cuda())
    # device = torch.device("cuda")
    # npoint = 1024
    # testDataLoader = torch.utils.data.DataLoader(ModelNet40(partition='test', num_points=1024), num_workers=8,
    #                         batch_size=4, shuffle=True, drop_last=False)
    # for data, label in testDataLoader:
    #     data, label = data.to(device), label.to(device).squeeze()
    #     B, N, C = data.shape
    #     #wrs sample
    #     centroids,idx = weighted_random_point_sample(data, npoint, k=20) #(B,N)
    #     print("data:",data.shape)
    #     print("cent:",centroids.shape)
    #     new_data = torch.gather(data, 1, centroids.unsqueeze(-1).expand(B, npoint, C))#[B,N,3]
    #     print("new_data:",new_data.shape)

    #     normals = torch.zeros_like(new_data).to(device)#[B,N,3]
    #     new_pts = torch.zeros_like(new_data).to(device)#[B,N,3]
    #     pts = torch.zeros_like(data).to(device)#[B,N,3]
    #     for i in range(new_data.shape[0]):
    #         normals[i] = I.get_normal_vector(new_data[i].unsqueeze(0)).squeeze(0)
    #         print("----normal----",normals.dtype)
    #         print("-----newdata---",new_data.dtype)
    #         new_pts[i] = I.shape_invariant_perturb(new_data[i].unsqueeze(0), normals[i].unsqueeze(0)).squeeze(0)
    #     #     pts[i] = I.shape_invariant_perturb(data[i].unsqueeze(0), normals[i].unsqueeze(0)).squeeze(0)
    #     pt = I.random_k_neighbors_shape_invariant_perturb(data, idx, normals, threshold=0.03)
    #     # new_pts = new_pts.detach().cpu().numpy()
    #     # print("normals:",normals.shape)
    #     # print("new_pts:",new_pts.shape)
    #     print("pt",pt.shape)
    #     break
        