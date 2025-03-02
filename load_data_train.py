import os,json
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
def get_data(splits = ['train','val','test']):
    basedir = os.path.join('data/nerf_synthetic','lego')
    metas = {}
    for s in splits:
        with open(os.path.join(basedir,f'transforms_{s}.json'),'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits :
        meta = metas[s]
        imgs = []
        poses = []
        # 取出frame下的所有key
        for frame in meta['frames'][::]:
            fname = os.path.join(basedir,frame['file_path']+'.png')
            imgs.append(imageio.imread(fname)) # 4-d
            poses.append(np.array(frame['transform_matrix']))
        
        # rgba图像归一化，a代表不透明度 并且通通变成ndarry
        imgs = (np.array(imgs)/255).astype(np.float32)
        poses = np.array(poses).astype(np.float32)

        # 累计计数?
        counts.append(counts[-1]+imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    # 不同数据集里面的图像个数不一样，所以all_imgs列表是不均匀排布的，不能直接np.array
    imgs = np.concatenate(all_imgs,0) # 注意这里的图像是4通道的
    poses = np.concatenate(all_poses,0) # poses是相机外参
    '''
    plt.figure('lego')
    plt.imshow(imgs[10])
    plt.title('lego_image')
    plt.show()
    '''
    # 各个数据集角度都差不多，算一个
    camera_angle = meta['camera_angle_x']
    H,W = imgs[0].shape[0:2]
    f = 0.5*W / np.tan(0.5*camera_angle)
    intrinsic_matrix = np.array([[f,0,0.5*W],
                                [0,f,0.5*H],
                                [0,0,1]])
    
    
    return imgs,poses,intrinsic_matrix

def get_rays_np(H, W, K, c2w):
    '''
    c2w = [ R_3x3, t_3x1
            0_1x3, 1    ]
    前3行3列表示旋转
    t表示平移
    '''

    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')

    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1) # [H, W, 3]
    # [3,3] * [H, W, 1, 3] 对应元素相乘，再求和 <=> c2w @ dir in dirs
    rays_d = np.sum(c2w[:3,:3] * dirs[..., np.newaxis, :], -1)  # [H, W, 3]
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True) # 转换成方向向量
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d)) # [H, W, 3]

    return rays_o, rays_d

# def get_rays_rgb(splits = ['train','val','test']):
#     imgs, poses, K = get_data(splits)
#     # 4通道转白色背景的3通道
#     imgs = imgs[...,:3]*imgs[...,-1:] + (1.-imgs[...,-1:])
#     H,W = imgs.shape[1:3]

#     rays = np.stack([get_rays_np(H, W, K, pose) for pose in poses[:,:3,:4]], axis=0)

#     rays_rgb = np.concatenate([rays, imgs[: ,None]], axis=1) # [num_pic, ro_rd_rgb, H, W, 3]

#     rays_rgb = np.transpose(rays_rgb,[0, 2, 3, 1, 4]) # [num_pic, H, W, ro_rd_rgb, 3]

#     return rays_rgb, imgs, poses

if __name__ == '__main__':
    splits = ['train','val']
    name = 'lego'
    n = len(splits)
    for i in range(n):
        name += '_'
        name += splits[i]
    
    imgs, poses, K = get_data(splits)
    images = imgs[...,:3]*imgs[...,-1:] + (1.-imgs[...,-1:])
    focal = K[0,0]
    init_o = K[:3,-1]

    SAVE = True
    if SAVE:
        print('Start saving data...')
        np.savez_compressed(name,images=images,poses=poses,focal=K[0,0])
        print('Data saved!')
