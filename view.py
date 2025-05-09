import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

MAP = ['uniform',
        'gaussian',
       'background',
       'impulse',
    #    'scale',
       'upsampling',
       'shear',
       'rotation',
       'cutout',
       'density',
       'density_inc',
       'distortion',
       'distortion_rbf',
       'distortion_rbf_inv',
       'occlusion',
       'lidar',
       'original'
]
# 创建一个3D图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# 设置图像的标题和坐标轴标签
ax.set_title('Point Cloud Visualization')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 加载点云数据
for cor in MAP:
    if cor in ['original']:
        data_path = f'./data/modelnet40_c/data_{cor}.npy'
        data = np.load(data_path)
        # data = np.load('/home/user_tp/workspace/code/ModelNet40-C/data/modelnet40_c/data_gaussian_1.npy')

        # 随机选择一个样本进行可视化
        # sample_id = np.random.randint(0, 2468)
        sample_id = 1234
        sample_data = data[sample_id]
        # 将点云数据添加到图像中
        ax.scatter(sample_data[:, 0], sample_data[:, 1], sample_data[:, 2], c='r')
        # 将图像保存到文件中
        plt.savefig(f'./image/data_{cor}.png')
    else:
        for i in range(5):
            data_path = f'./data/modelnet40_c/data_{cor}_{i+1}.npy'
            data = np.load(data_path)
        # data = np.load('/home/user_tp/workspace/code/ModelNet40-C/data/modelnet40_c/data_gaussian_1.npy')

        # 随机选择一个样本进行可视化
        # sample_id = np.random.randint(0, 2468)
            sample_id = 1234
            sample_data = data[sample_id]
        # 将点云数据添加到图像中
            ax.scatter(sample_data[:, 0], sample_data[:, 1], sample_data[:, 2], c='r')
        # 将图像保存到文件中
            plt.savefig(f'./image/data_{cor}_{i+1}.png')

# 显示图像
# plt.show()