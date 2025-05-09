""" Utility functions for processing point clouds.
Author: Charles R. Qi, Hao Su
Date: November 2016
"""
###from saliency/pointnet-master/utils/pc_util.py
import os
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, sys.path[0]+"/../")
# from data import load_data_h5
from pointnet2_ops import pointnet2_utils
from sampling import weighted_random_point_sample, knn,cal_weight,downsample_point_cloud_knn,downsample_point_cloud
from torch.utils.data import DataLoader
# from util import fps,wfps,wfarthest_point_sample
import torch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
# Draw point cloud
from eulerangles import euler2mat
# Point cloud IO
import numpy as np
from plyfile import PlyData, PlyElement
from PIL import Image
# ----------------------------------------
# Point Cloud/Volume Conversions
# ----------------------------------------
def point_cloud_to_volume_batch(point_clouds, vsize=12, radius=1.0, flatten=True):
    """ Input is BxNx3 batch of point cloud
        Output is Bx(vsize^3)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume(np.squeeze(point_clouds[b,:,:]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)

def point_cloud_to_volume(points, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize,vsize,vsize))
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel
    locations = locations.astype(int)
    vol[locations[:,0],locations[:,1],locations[:,2]] = 1.0
    return vol

#a = np.zeros((16,1024,3))
#print point_cloud_to_volume_batch(a, 12, 1.0, False).shape

def volume_to_point_cloud(vol):
    """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
        return Nx3 numpy array.
    """
    vsize = vol.shape[0]
    assert(vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a,b,c] == 1:
                    points.append(np.array([a,b,c]))
    if len(points) == 0:
        return np.zeros((0,3))
    points = np.vstack(points)
    return points

def point_cloud_to_volume_v2_batch(point_clouds, vsize=12, radius=1.0, num_sample=128):
    """ Input is BxNx3 a batch of point cloud
        Output is BxVxVxVxnum_samplex3
        Added on Feb 19
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume_v2(point_clouds[b,:,:], vsize, radius, num_sample)
        vol_list.append(np.expand_dims(vol, 0))
    return np.concatenate(vol_list, 0)

def point_cloud_to_volume_v2(points, vsize, radius=1.0, num_sample=128):
    """ input is Nx3 points
        output is vsize*vsize*vsize*num_sample*3
        assumes points are in range [-radius, radius]
        samples num_sample points in each voxel, if there are less than
        num_sample points, replicate the points
        Added on Feb 19
    """
    vol = np.zeros((vsize,vsize,vsize,num_sample,3))
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel
    locations = locations.astype(int)
    loc2pc = {}
    for n in range(points.shape[0]):
        loc = tuple(locations[n,:])
        if loc not in loc2pc:
            loc2pc[loc] = []
        loc2pc[loc].append(points[n,:])
    #print loc2pc

    for i in range(vsize):
        for j in range(vsize):
            for k in range(vsize):
                if (i,j,k) not in loc2pc:
                    vol[i,j,k,:,:] = np.zeros((num_sample,3))
                else:
                    pc = loc2pc[(i,j,k)] # a list of (3,) arrays
                    pc = np.vstack(pc) # kx3
                    # Sample/pad to num_sample points
                    if pc.shape[0]>num_sample:
                        choices = np.random.choice(pc.shape[0], num_sample, replace=False)
                        pc = pc[choices,:]
                    elif pc.shape[0]<num_sample:
                        pc = np.lib.pad(pc, ((0,num_sample-pc.shape[0]),(0,0)), 'edge')
                    # Normalize
                    pc_center = (np.array([i,j,k])+0.5)*voxel - radius
                    #print 'pc center: ', pc_center
                    pc = (pc - pc_center) / voxel # shift and scale
                    vol[i,j,k,:,:] = pc 
                #print (i,j,k), vol[i,j,k,:,:]
    return vol

def point_cloud_to_image_batch(point_clouds, imgsize, radius=1.0, num_sample=128):
    """ Input is BxNx3 a batch of point cloud
        Output is BxIxIxnum_samplex3
        Added on Feb 19
    """
    img_list = []
    for b in range(point_clouds.shape[0]):
        img = point_cloud_to_image(point_clouds[b,:,:], imgsize, radius, num_sample)
        img_list.append(np.expand_dims(img, 0))
    return np.concatenate(img_list, 0)

def point_cloud_to_image(points, imgsize, radius=1.0, num_sample=128):
    """ input is Nx3 points
        output is imgsize*imgsize*num_sample*3
        assumes points are in range [-radius, radius]
        samples num_sample points in each pixel, if there are less than
        num_sample points, replicate the points
        Added on Feb 19
    """
    img = np.zeros((imgsize, imgsize, num_sample, 3))
    pixel = 2*radius/float(imgsize)
    locations = (points[:,0:2] + radius)/pixel # Nx2
    locations = locations.astype(int)
    loc2pc = {}
    for n in range(points.shape[0]):
        loc = tuple(locations[n,:])
        if loc not in loc2pc:
            loc2pc[loc] = []
        loc2pc[loc].append(points[n,:])
    for i in range(imgsize):
        for j in range(imgsize):
            if (i,j) not in loc2pc:
                img[i,j,:,:] = np.zeros((num_sample,3))
            else:
                pc = loc2pc[(i,j)]
                pc = np.vstack(pc)
                if pc.shape[0]>num_sample:
                    choices = np.random.choice(pc.shape[0], num_sample, replace=False)
                    pc = pc[choices,:]
                elif pc.shape[0]<num_sample:
                    pc = np.lib.pad(pc, ((0,num_sample-pc.shape[0]),(0,0)), 'edge')
                pc_center = (np.array([i,j])+0.5)*pixel - radius
                pc[:,0:2] = (pc[:,0:2] - pc_center)/pixel
                img[i,j,:,:] = pc
    return img
# ----------------------------------------
# Point cloud IO
# ----------------------------------------
def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array

def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)
# ----------------------------------------
# Simple Point cloud and Volume Renderers
# ----------------------------------------
###changed 
def draw_point_cloud_sampled(input_points,colors, canvasSize=500, space=200, diameter=25,
                     xrot=0, yrot=0, zrot=0, switch_xyz=[0,1,2], normalize=True):
    """ Render point cloud to image with alpha channel.
        Input:
            input_points: Nx3 numpy array (x, y, z)
            colors: sample index points
        Output:
            RGB image as numpy array of size canvasSizexcanvasSize
    """
    # image = np.zeros((canvasSize, canvasSize, 3))
    image = np.ones((canvasSize, canvasSize, 3)) * 255  # 初始化为白色
    if input_points is None or input_points.shape[0] == 0:
        return image
    points = input_points[:, switch_xyz]
    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # Normalize the point cloud
    # We normalize scale to fit points in a unit sphere
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))
        points /= furthest_distance
    # Pre-compute the Gaussian disk
    radius = (diameter-1)/2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i-radius) + (j-radius) * (j-radius) <= radius * radius:
                disk[i, j] = np.exp((-(i-radius)**2 - (j-radius)**2)/(radius**2))
    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]

    # Order points by z-buffer
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    colors = colors[zorder, :]

    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSize/2 + (x*space)
        yc = canvasSize/2 + (y*space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))

        px = dx + xc
        py = dy + yc
        image[px, py, :] = dv[:, np.newaxis] * colors[j, :]
    # image = image / np.max(image)
    image = np.clip(image, 0, 1)
    return image

def draw_point_cloud(input_points, canvasSize=500, space=200, diameter=25,
                     xrot=0, yrot=0, zrot=0, switch_xyz=[0,1,2], normalize=True):
    """ Render point cloud to image with alpha channel.
        Input:
            points: Nx3 numpy array (+y is up direction)
        Output:
            gray image as numpy array of size canvasSizexcanvasSize
    """
    image = np.zeros((canvasSize, canvasSize))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]
    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # Normalize the point cloud
    # We normalize scale to fit points in a unit sphere
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
        points /= furthest_distance

    # Pre-compute the Gaussian disk
    radius = (diameter-1)/2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i-radius) + (j-radius) * (j-radius) <= radius * radius:
                disk[i, j] = np.exp((-(i-radius)**2 - (j-radius)**2)/(radius**2))
    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]
    
    # Order points by z-buffer
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
    max_depth = np.max(points[:, 2])
       
    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSize/2 + (x*space)
        yc = canvasSize/2 + (y*space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))
        
        px = dx + xc
        py = dy + yc
        
        image[px, py] = image[px, py] * 0.7 + dv * (max_depth - points[j, 2]) * 0.3
    
    image = image / np.max(image)
    return image

def point_cloud_one_demo(point,colors,save_path):
    im_array = draw_point_cloud_sampled(point, colors, zrot=90/180.0*np.pi, xrot=0/180.0*np.pi, yrot=45/180.0*np.pi,diameter=12)
    img = Image.fromarray(np.uint8(im_array*255.0))
    img.save(save_path+'.jpg')

def point_cloud_three_views(points,colors):
    """ input points Nx3 numpy array (+y is up direction).
        return an numpy array gray image of size 500x1500. """ 
    # +y is up direction
    # xrot is azimuth
    # yrot is in-plane
    # zrot is elevation
    img1 = draw_point_cloud_sampled(points, colors[0], zrot=75/180.0*np.pi, xrot=90/180.0*np.pi, yrot=0/180.0*np.pi,diameter=12)
    img2 = draw_point_cloud_sampled(points, colors[1], zrot=75/180.0*np.pi, xrot=90/180.0*np.pi, yrot=0/180.0*np.pi,diameter=12)#70,135,0
    img3 = draw_point_cloud_sampled(points, colors[2], zrot=75/180.0*np.pi, xrot=90/180.0*np.pi, yrot=0/180.0*np.pi,diameter=12)#180,90,0
    image_lar = np.concatenate([img1, img2, img3], 1)
    return image_lar

def point_cloud_three_views_demo(points,colors,save_path):
    """ Demo for draw_point_cloud function """
    # points = read_ply('/home/user_tp/workspace/data/ModelNet40/airplane/train/airplane_0002.ply')
    im_array = point_cloud_three_views(points, colors)
    img = Image.fromarray(np.uint8(im_array*255.0))
    img.save(save_path+".jpg")

def pyplot_draw_point_cloud(points, output_filename):
    """ points is a Nx3 numpy array """
   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], s=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig(output_filename+'image.jpg')
    plt.close()
    
def pyplot_draw_point_cloud_nat_and_adv(points, points_adv, output_filename):
    """ points is a Nx3 numpy array """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    fig = plt.figure()
    xmin, xmax = np.min(points[:,0])-0.1, np.max(points[:,0])+0.1
    ymin, ymax = np.min(points[:,1])-0.1, np.max(points[:,1])+0.1
    zmin, zmax = np.min(points[:,2])-0.1, np.max(points[:,2])+0.1
    
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,1], points[:,2], points[:,0], s=5, c='b')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_zlabel('x')
    ax.set_xlim(ymin, ymax)
    ax.set_ylim(zmin, zmax)
    ax.set_zlim(xmin, xmax)
    plt.axis('off')
    plt.savefig(output_filename+'_1nat_x.jpg')
    plt.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_adv[:,1], points_adv[:,2], points_adv[:,0], s=5, c='r')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_zlabel('x')
    ax.set_xlim(ymin, ymax)
    ax.set_ylim(zmin, zmax)
    ax.set_zlim(xmin, xmax)
    plt.axis('off')
    plt.savefig(output_filename+'_2adv_x.jpg')
    plt.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,2], points[:,0], points[:,1], s=5, c='b')
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(xmin, xmax)
    ax.set_zlim(ymin, ymax)
    plt.axis('off')
    plt.savefig(output_filename+'_3nat_y.jpg')
    plt.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_adv[:,2], points_adv[:,0], points_adv[:,1], s=5, c='r')
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(xmin, xmax)
    ax.set_zlim(ymin, ymax)
    plt.axis('off')
    plt.savefig(output_filename+'_4adv_y.jpg')
    plt.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], s=5, c='b')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    plt.axis('off')
    plt.savefig(output_filename+'_5nat_z.jpg')
    plt.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_adv[:,0], points_adv[:,1], points_adv[:,2], s=5, c='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    plt.axis('off')
    plt.savefig(output_filename+'_6adv_z.jpg')
    plt.close()

def plot_nat_interval_adv(points, drop_points, image_filename):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    xmin, xmax = np.min(points[:,0])-0.1, np.max(points[:,0])+0.1
    ymin, ymax = np.min(points[:,1])-0.1, np.max(points[:,1])+0.1
    zmin, zmax = np.min(points[:,2])-0.1, np.max(points[:,2])+0.1
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,2], points[:,0], points[:,1], s=1, c='k')
    ax.scatter(drop_points[:,2], drop_points[:,0], drop_points[:,1], s=1, c='k')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_zlabel('x')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(xmin, xmax)
    ax.set_zlim(ymin, ymax)
    plt.axis('off')
    plt.savefig(image_filename+'_y1.jpg', bbox_inches='tight')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,2], points[:,0], points[:,1], s=1, c='k')
    ax.scatter(drop_points[:,2], drop_points[:,0], drop_points[:,1], s=25, c='r')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_zlabel('x')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(xmin, xmax)
    ax.set_zlim(ymin, ymax)
    plt.axis('off')
    plt.savefig(image_filename+'_y2.jpg', bbox_inches='tight')
    plt.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,2], points[:,0], points[:,1], s=1, c='k')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_zlabel('x')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(xmin, xmax)
    ax.set_zlim(ymin, ymax)
    plt.axis('off')
    plt.savefig(image_filename+'_y3.jpg', bbox_inches='tight')
    plt.close()

def pyplot_draw_volume(vol, output_filename):
    """ vol is of size vsize*vsize*vsize
        output an image to output_filename
    """
    points = volume_to_point_cloud(vol)
    pyplot_draw_point_cloud(points, output_filename)

def write_ply_color(points, labels, out_filename, num_classes=None):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    import matplotlib.pyplot as pyplot
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels)+1
    else:
        assert(num_classes>np.max(labels))
    fout = open(out_filename, 'w')
    #colors = [pyplot.cm.hsv(i/float(num_classes)) for i in range(num_classes)]
    colors = [pyplot.cm.jet(i/float(num_classes)) for i in range(num_classes)]
    for i in range(N):
        c = colors[labels[i]]
        c = [int(x*255) for x in c]
        fout.write('v %f %f %f %d %d %d\n' % (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]))
    fout.close()

###add
def read_txt(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            # 假设点的坐标是以空格分隔的
            coordinates = [float(coord) for coord in line.strip().split(',')[:3]]
            points.append(coordinates)
    return np.array(points)

#Pointnet_Pointnet2_pytorch/data_utils/ModelNetDataLoader.py
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    centroids = centroids.astype(np.int32)
    point = point[centroids]
    return point,centroids

###add
def set_color(point,npoint):
    color = np.ones_like(point) * [0, 1, 0]#设置点云颜色
    colors = np.repeat(color[np.newaxis, ...], 3, axis=0)
    # _,centroid1 = farthest_point_sample(point, npoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    point = np.expand_dims(point, axis=0)
    points = torch.tensor(point, dtype=torch.float32).to(device)
    centroid1 = fps(points, npoint).squeeze().cpu().numpy().astype(np.int32)
    centroid2,_ = weighted_random_point_sample(points, npoint, k=20).cpu().numpy().astype(np.int32)#random sample or wrs
    weights = cal_weight(points, k=20)
    # centroid2 = wfarthest_point_sample(points, weights, npoint).long().squeeze().cpu().numpy().astype(np.int32)
    centroid3 = pointnet2_utils.wfurthest_point_sample(points, weights*1e2, npoint).long().squeeze().cpu().numpy().astype(np.int32)
    # centroid3,_ = weighted_random_point_sample(points, npoint, k=16).cpu().numpy().astype(np.int32)
    # print("centroid3:",centroid3)
    centroids = np.stack([centroid1, centroid2, centroid3], axis=0)
    for i in range(colors.shape[0]):
        colors[i, centroids[i], :] = [1, 0, 0]#设置采样点颜色
    return colors

###add
def draw(point_cloud, save_path):
    _, centroids = farthest_point_sample(point, 512)
    colors = np.full((1024, 3), [0, 1, 0], dtype=float)
    # 将索引对应的点标记为红色
    colors[centroids.astype(np.int32)] = [1, 0, 0]
    # plt.scatter(point[:, 0], point[:, 1], c=colors, marker='.', s=50)  # s 参数控制点的大小
    # plt.title('Point Cloud Visualization')
    # plt.axis('off')
    # 可视化点云
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=colors, marker='.')
    ax.set_axis_off()
    # 显示图形
    plt.show()
    plt.savefig(save_path+'show.jpg')

###add
def draw_noise(data_path, dataset, path, npoint):
    MAPC = ['uniform', 'gaussian', 'background', 'impulse', 'upsampling', 'shear', 'rotation', 'cutout',
       'density', 'density_inc', 'distortion', 'distortion_rbf', 'distortion_rbf_inv', 'occlusion', 'lidar',]
    mapc = ['add_global', 'add_local', 'dropout_global', 'dropout_local', 'jitter', 'rotate', 'scale']
    if dataset == "MNC":
        for cor in MAPC:
            data_root = f"{data_path}data_{cor}_1.npy"
            point_set = np.load(data_root, allow_pickle=True)
            #set colors
            colors = set_color(point_set[120],npoint)
            save_path =  f"{path}ModelNet40-C/120/perb/{cor}.jpg"
            ###three demos
            point_cloud_three_views_demo(point_set[120],colors, save_path)
            ###one demo
    elif dataset == "mnc":
        for cor in mapc:
            data_root = f"{data_path}{cor}_0.h5"
            point_set, label = load_data_h5(data_root)
            #set colors
            colors = set_color(point_set[125],npoint)
            save_path =  f"{path}modelnet_c/125/mean/{cor}.jpg"
            point_cloud_three_views_demo(point_set[125],colors, save_path)
    """ for i in range(point_set.shape[0]):
        colors = set_color(point_set[i],npoint)
        save_path =  f"{path}{i}.jpg"
        point_cloud_three_views_demo(point_set[i],colors, save_path) """
def draw_mn(args, device, i):
    step_size1=0.02
    step_size2=0.03
    data_path = '/home/user_tp/workspace/data/modelnet40_normal_resampled/'
    testDataLoader = DataLoader(ModelNetDataLoader(root=data_path, args=args, split='test'), 
                             batch_size=1, shuffle=True, num_workers=10)

    # testDataLoader = torch.utils.data.DataLoader(ModelNet40(partition='test', num_points=1024), num_workers=8,
                            # batch_size=1, shuffle=True, drop_last=False)
    for data, label in testDataLoader:
        data, label = data.to(device), label.to(device).squeeze()
        B, N, C = data.shape
        save_path = "/home/user_tp/workspace/code/base_model/PCT_Pytorch-main/visual/images/"
        point = data #mn
        print("point:",point.shape)
        # point,_ = farthest_point_sample(pc, 1024) #[N,3] dtype:torch.float32
        npoint = point.shape[0]
        # colors = set_color(point,npoint) #fps wrs fwrs sampled
        save =  f"{save_path}modelnet40/"
        if not os.path.exists(save):
            os.makedirs(save)
        # save_demo =  f"{save}demo/"
        save_plt1 =  f"{save}origin_{step_size1}/"
        if not os.path.exists(save_plt1):
            os.makedirs(save_plt1)
        save_plt2 =  f"{save}origin_{step_size2}/"
        if not os.path.exists(save_plt2):
            os.makedirs(save_plt2)
        # pyplot_draw_point_cloud(point, save_path1)
        # point_cloud_three_views_demo(point,colors, save_path)
        device = torch.device("cuda")
        point = point.squeeze(0).detach().cpu().numpy()
        points = torch.from_numpy(point).to(device).to(torch.float32)#[N,3]
        # points = point.squeeze(0)
        N,C = points.shape


        I1 = Interpolation(step_size1)
        ###origin
        normals = I1.get_normal_vector(points.unsqueeze(0)).squeeze(0)#[N,3] dtype torch.float32
        pts_p = I1.shape_invariant_perturb(points.unsqueeze(0), normals.unsqueeze(0)).squeeze(0)
        # _, idx = knn(points.unsqueeze(0), k=20)
        # pts_p = I1.random_k_neighbors_shape_invariant_perturb(points.unsqueeze(0), idx, normals.unsqueeze(0)).squeeze(0)
        print("pts_p:",pts_p.shape)
        # pts = pts.detach().cpu().numpy()
        pts = torch.cat((points,pts_p),dim=0).detach().cpu().numpy()
        pts_p = pts_p.detach().cpu().numpy()
        #plt
        pyplot_draw_point_cloud(point, f"{save_plt1}origin_1024_{i}")
        pyplot_draw_point_cloud(pts, f"{save_plt1}origin_2048_{i}")
        pyplot_draw_point_cloud(pts_p, f"{save_plt1}operb_1024_{i}")

        #step_size2=0.03
        I2 = Interpolation(step_size2)
        ###origin
        normals2 = I2.get_normal_vector(points.unsqueeze(0)).squeeze(0)#[N,3] dtype torch.float32
        # pts_p = I.shape_invariant_perturb(points.unsqueeze(0), normals.unsqueeze(0)).squeeze(0)
        _, idx = knn(points.unsqueeze(0), k=20)
        pts_rp = I2.random_k_neighbors_shape_invariant_perturb(points.unsqueeze(0), idx, normals2.unsqueeze(0)).squeeze(0)
        print("pts_p:",pts_rp.shape)
        # pts = pts.detach().cpu().numpy()
        ptsr = torch.cat((points,pts_rp),dim=0).detach().cpu().numpy()
        pts_rp = pts_rp.detach().cpu().numpy()
        #plt
        pyplot_draw_point_cloud(point, f"{save_plt2}origin_1024_{i}")
        pyplot_draw_point_cloud(ptsr, f"{save_plt2}origin_2048_{i}")
        pyplot_draw_point_cloud(pts_rp, f"{save_plt2}operb_1024_{i}")
        break

if __name__=="__main__":
    # from interpolation import Interpolation,knn
    # 获取当前文件（pc.py）的绝对路径  
    current_file_path = os.path.abspath(__file__)  
    # 获取当前文件的目录  
    current_directory = os.path.dirname(current_file_path)  
    # 获取MNC/dgcnn/pyt/的绝对路径  
    dgcnn_pyt_directory = os.path.abspath(os.path.join(current_directory, '../../dgcnn/pytorch'))  
    
    # 将MNC/dgcnn/pyt/目录添加到sys.path中  
    sys.path.append(dgcnn_pyt_directory)
    from data import load_data,ModelNet40
    import argparse

    # parser = argparse.ArgumentParser('training')
    # parser.add_argument('--batch_size', type=int, default=4, help='batch size in training')
    # parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    # parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    # parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    # parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    # parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    # parser.add_argument('--dataset', type=str, default='MNC', help='for the dataset,MNC is modelnet-c, mnc is pointcloud-c')
    # args = parser.parse_args()
    # points = read_ply('/home/user_tp/workspace/data/ModelNet40/airplane/train/airplane_0001.ply')
    # points,_ = farthest_point_sample(points, 1024)
    device = torch.device("cuda")
    npoint = 1024
    step_size = 0.03
    m = 2
    i = '-1'
    # draw_mn(args, device,i)#干净数据集
    # data_path = "/home/user_tp/workspace/code/attack/ModelNet40-C/data/modelnet40_ply_hdf5_2048/"
    # testDataLoader = DataLoader(ModelNet40(num_points=1024, partition='test'),
    #                             batch_size=1, shuffle=True, num_workers=10)

    # testDataLoader = torch.utils.data.DataLoader(ModelNet40(partition='test', num_points=1024), num_workers=8,
                            # batch_size=1, shuffle=True, drop_last=False)
    for data, label in testDataLoader:
        data, label = data.to(device), label.to(device).squeeze()
    
        print("data.shape",data.shape)
        r = torch.normal(mean=torch.tensor([1.0]), std=torch.tensor([0.2]))  # Gaussian distribution with mean=1.0, std=0.1
        ratio = torch.clamp(r, 0.75, 1.0)  # Clip ratio to be within [0.75, 1.0]
        _, idx_k = knn(data, k=256)
        p_cloud = downsample_point_cloud(data, ratio)
        p_cloud_k = downsample_point_cloud_knn(data, idx_k,ratio)
        print("p_cloud:",p_cloud.shape)
        print("p_cloud_k:",p_cloud_k.shape)


    # n=2460
    # # # pyplot_draw_point_cloud(points, save_path)
    # save_path = "/home/user_tp/workspace/code/base_model/PCT_Pytorch-main/visual/images/"
    # #mnc
    # sev='uniform'
    # data_path = "/home/user_tp/workspace/data/ModelNet40-C/"
    # data_root = f"{data_path}data_{sev}_5.npy"
    # point_set = np.load(data_root, allow_pickle=True)
    # save =  f"{save_path}ModelNet40-C/randomk/{sev}/"
    # #pcc
    # # sev='scale'
    # # data_path = "/home/user_tp/workspace/data/modelnet_c/"
    # # data_root = f"{data_path}{sev}_2.h5"#pcc
    # # point_set, label = load_data_h5(data_root)
    # # save =  f"{save_path}modelnet_c/randomk/{sev}/"
    # point = point_set[n]#mnc or pcc
    # # point = point_set[n] #mn
    # print("point:",point.shape)
    # # point,_ = farthest_point_sample(pc, 1024) #[N,3] dtype:torch.float32
    # npoint = point.shape[0]
    # # colors = set_color(point,npoint) #fps wrs fwrs sampled
    # if not os.path.exists(save):
    #     os.makedirs(save)
    # # save_demo =  f"{save}demo/"
    # save_plt =  f"{save}{n}/"
    # if not os.path.exists(save_plt):
    #     os.makedirs(save_plt)
    # # pyplot_draw_point_cloud(point, save_path1)
    # # point_cloud_three_views_demo(point,colors, save_path)
    # device = torch.device("cuda")
    # points = torch.from_numpy(point).to(device).to(torch.float32)#[N,3]
    # N,C = points.shape
    # # points = points.detach().cpu().numpy()
    # I = Interpolation(step_size)

    # ###origin
    # normals = I.get_normal_vector(points.unsqueeze(0)).squeeze(0)#[N,3] dtype torch.float32
    # # pts_p = I.shape_invariant_perturb(points.unsqueeze(0), normals.unsqueeze(0)).squeeze(0)
    # _, idx = knn(points.unsqueeze(0), k=20)
    # pts_p = I.random_k_neighbors_shape_invariant_perturb(points.unsqueeze(0), idx, normals.unsqueeze(0)).squeeze(0)
    
    # print("pts_p:",pts_p.shape)
    # # pts = pts.detach().cpu().numpy()
    # pts = torch.cat((points,pts_p),dim=0).detach().cpu().numpy()
    # pts_p = pts_p.detach().cpu().numpy()

    # #plt
    # pyplot_draw_point_cloud(point, f"{save_plt}origin_1024_{step_size}_{m}")
    # pyplot_draw_point_cloud(pts, f"{save_plt}ro_2048_{step_size}_{m}")
    # pyplot_draw_point_cloud(pts_p, f"{save_plt}orperb_1024_{step_size}_{m}")

    # #demo
    # # color1 = np.ones_like(point) * [0, 1, 0]#设置点云颜色 no sample
    # # point_cloud_one_demo(point,color1, save_demo+'origin_1024')#[N,3] 1024
    # # color2 = np.ones_like(pts) * [0, 1, 0]#设置点云颜色 no sample
    # # point_cloud_one_demo(pts,color2, save_demo+'origin_2048')#[N,3] 2048

    # # ####wrs sample #replacement=false
    # # centroids_f,idx = weighted_random_point_sample(points.unsqueeze(0), npoint, k=20) #(B,N)
    # # new_data_f = torch.gather(points.unsqueeze(0), 1, centroids_f.unsqueeze(-1).expand(1, npoint, C)).squeeze(0)
    # # print("new_data:",new_data_f.shape)
    # # normals_f = I.get_normal_vector(new_data_f.unsqueeze(0)).squeeze(0)
    # # data_f = I.shape_invariant_perturb(new_data_f.unsqueeze(0), normals_f.unsqueeze(0)).squeeze(0)
    # # pts_f = torch.cat((points,data_f),dim=0).detach().cpu().numpy()
    # # ndata_f = new_data_f.detach().cpu().numpy()

    # ###wrs #replacement = true
    # centroids = torch.zeros(1, npoint, dtype=torch.long)#.to(device)
    # centroids_r,_ = weighted_random_point_sample(points.unsqueeze(0), npoint, k=20,replace=True) #(B,N)
    # new_data_r = torch.gather(points.unsqueeze(0), 1, centroids_r.unsqueeze(-1).expand(1, npoint, C)).squeeze(0)
    # print("new_data:",new_data_r.shape)
    # normals_r = I.get_normal_vector(new_data_r.unsqueeze(0)).squeeze(0)
    # # data_r = I.shape_invariant_perturb(new_data_r.unsqueeze(0), normals_r.unsqueeze(0)).squeeze(0)
    # _, idx = knn(points.unsqueeze(0), k=20)
    # data_r = I.random_k_neighbors_shape_invariant_perturb(points.unsqueeze(0), idx, normals_r.unsqueeze(0)).squeeze(0)
    
    # pts_r = torch.cat((points,data_r),dim=0).detach().cpu().numpy()
    # ndata_r = new_data_r.detach().cpu().numpy()
    # data_r = data_r.detach().cpu().numpy()

    # #plt
    # # pyplot_draw_point_cloud(ndata_f, f"{save_plt}wrs_1024_{step_size}")
    # # pyplot_draw_point_cloud(pts_f, f"{save_plt}wrs_2048_{step_size}")
    # pyplot_draw_point_cloud(ndata_r, f"{save_plt}rwrs_1024_{step_size}_{m}")
    # pyplot_draw_point_cloud(pts_r, f"{save_plt}rrwrs_2048_{step_size}_{m}")
    # pyplot_draw_point_cloud(data_r, f"{save_plt}rwrperb_1024_{step_size}_{m}")

    # demo
    # color3 = np.ones_like(ndata_f) * [0, 1, 0]
    # point_cloud_one_demo(ndata_f,color3, save_demo+'wrs_1024')#[N,3] 1024
    # color4 = np.ones_like(pts_f) * [0, 1, 0]#设置点云颜色 no sample
    # point_cloud_one_demo(pts_f,color4, save_demo+'wrs_2048')#[N,3] 2048

    #噪声
    # data_path = "/home/user_tp/workspace/data/modelnet_c/"
    # data_path = "/home/user_tp/workspace/data/ModelNet40-C/"
    # dataset = "MNC"# MNC or mnc
    # draw_noise(data_path, dataset, save_path, npoint)
    # draw(point, save_path)
    # from sampling import weighted_random_point_sample