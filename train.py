import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import os
from tqdm import trange
from helper import *

# Set seed.
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
# Initialize coarse and fine MLPs.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
F_c = NeRF().to(device)
F_f = NeRF().to(device)
# Number of query points passed through the MLP at a time. See: https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf.py#L488.
chunk_size = 1024 * 32
# Number of training rays per iteration. See Section 5.3.
batch_img_size = 64
n_batch_pix = batch_img_size ** 2
# Initialize optimizer. See Section 5.3.
lr = 5e-4
optimizer = optim.Adam(list(F_c.parameters()) + list(F_f.parameters()), lr=lr)
criterion = nn.MSELoss()
# The learning rate decays exponentially. See Section 5.3
# See: https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf.py#L486.
lrate_decay = 250
decay_steps = lrate_decay * 1000
# See: https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf.py#L707.
decay_rate = 0.1

# Initialize volume rendering hyperparameters.
# Near bound. See Section 4.
t_n = 1.0
# Far bound. See Section 4.
t_f = 4.0
# Number of coarse samples along a ray. See Section 5.3.
N_c = 64
# Number of fine samples along a ray. See Section 5.3.
N_f = 128
# Bins used to sample depths along a ray. See Equation (2) in Section 4.
t_i_c_gap = (t_f - t_n) / N_c
t_i_c_bin_edges = (t_n + torch.arange(N_c) * t_i_c_gap).to(device)

# 导入数据
data_f = "lego_train_val.npz"
data = np.load(data_f)

# 展示图像
images = torch.tensor(data["images"].astype(np.float32))
if data_f == "car.npz":
  images = images/255
poses = torch.tensor(data["poses"].astype(np.float32))
focal = float(data['focal'])
# img_size = images.shape[1]
H, W = images.shape[1:3]

# 选择一张用来显示训练效果的图（图与位姿是一一对应的，从不同的位姿观察3D模型会产生不同的2D视图）
test_idx = 10
test_fig = plt.figure()
plt.imshow(images[test_idx])
test_img = torch.Tensor(images[test_idx]).to(device)

test_os, test_ds = get_rays(H, W, focal, poses[test_idx])
test_os = test_os.to(device) # [H, W, 3]
test_ds = test_ds.to(device)

# Start training model.
train_idxs = np.arange(len(images)) != test_idx
images = images[train_idxs]
poses = poses[train_idxs]
n_pix = H*W # 一张图总的像素数量

# 一个长为H*W，值为1/(H*W)的概率向量
pixel_ps = torch.full((n_pix,), 1 / n_pix).to(device)
psnrs = []
iternums = []
# See Section 5.3.
global_step = 0
num_iters = 100000
display_every = 1000
weight_every = 1000
loss_every = 100

logs_path = './logs'
if os.path.exists(logs_path) == False:
   os.mkdir(logs_path)
model_path = './model' # 训练模型的状态目录
if os.path.exists(model_path) == False:
   os.mkdir(model_path)

test_fig.savefig(logs_path + '/testfig.png')

# 检查点读取
LOAD_ckpts = True
if LOAD_ckpts:
  ckpts = [os.path.join(model_path,f) for f in sorted(os.listdir(model_path)) if 'tar' in f]
  # 升序排列，选择最后一轮开始训练
  if len(ckpts)>0:
    print('Found ckpts',f'-->{ckpts}')
    checkpoint = torch.load(ckpts[-1], weights_only=False)
    F_c.load_state_dict(checkpoint['model_coarse_state_dict'])
    F_f.load_state_dict(checkpoint['model_fine_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    global_step = checkpoint['global_step'] + 1
    psnrs = checkpoint['psnrs']
    iternums = checkpoint['iternums']
  else:
    print('No ckpts found')
    print('Train from scratch')

F_c.train()
F_f.train()

for i in trange(global_step, num_iters):
    # 从训练集中随机选一个位姿
    target_img_idx = np.random.randint(images.shape[0])
    target_pose = poses[target_img_idx].to(device)
    rays_o, rays_d = get_rays(H, W, focal, target_pose)

    # Sample a batch of rays.
    # 不放回的抽样，从pixel_ps的分布中抽出n_batch_pix个样本
    # 换言之，从所有的像素中抽出n_batch_pix个像素
    pix_idxs = pixel_ps.multinomial(n_batch_pix, False)
    # 将展平后的一维索引转换成原图片的二维索引
    pix_idx_rows = pix_idxs // W
    pix_idx_cols = pix_idxs % W

    # n_batch_pix = batch_img_size ** 2
    ds_batch = rays_d[pix_idx_rows, pix_idx_cols].reshape(
        batch_img_size, batch_img_size, -1
    )
    os_batch = rays_o[pix_idx_rows, pix_idx_cols].reshape(
        batch_img_size, batch_img_size, -1
    )

    # Run NeRF.
    (C_rs_c, C_rs_f) = run_one_iter_of_nerf(
        os_batch,
        ds_batch,
        t_i_c_bin_edges,
        t_i_c_gap,
        N_c,
        N_f,
        chunk_size,
        F_c,
        F_f,
    )
    target_img = images[target_img_idx].to(device)
    target_img_batch = target_img[pix_idx_rows, pix_idx_cols].reshape(C_rs_f.shape)
    # Calculate the mean squared error for both the coarse and fine MLP models and
    # update the weights. See Equation (6) in Section 5.3.
    loss = criterion(C_rs_c, target_img_batch) + criterion(C_rs_f, target_img_batch)
    if i % loss_every == 0 and i > 0:
        print(f"Iteration{i} --> Loss: {loss.item()}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Exponentially decay learning rate. See Section 5.3 and:
    # https://keras.io/api/optimizers/learning_rate_schedules/exponential_decay/.
    for g in optimizer.param_groups:
        g["lr"] = lr * decay_rate ** (i / decay_steps)

    if i % display_every == 0 and i > 0:
        F_c.eval()
        F_f.eval()
        with torch.no_grad():
            (_, C_rs_f) = run_one_iter_of_nerf(
                test_os,
                test_ds,
                t_i_c_bin_edges,
                t_i_c_gap,
                N_c,
                N_f,
                chunk_size,
                F_c,
                F_f,
            )

        loss = criterion(C_rs_f, test_img)
        psnr = -10.0 * torch.log10(loss)

        psnrs.append(psnr.item())
        iternums.append(i)

        target_fig = plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.imshow(C_rs_f.detach().cpu().numpy())
        plt.title(f"Iteration {i}")
        plt.subplot(122)
        plt.plot(iternums, psnrs)
        plt.title("PSNR")
        target_fig.savefig(logs_path + f'/Iteration{i}.png')

        F_c.train()
        F_f.train()

    # 存储模型，check points 检查点
    if i % weight_every == 0 and i > 0:
      path = os.path.join(model_path, '{:06d}.tar'.format(i))
      torch.save(
            {
                "model_coarse_state_dict": F_c.state_dict(),
                "model_fine_state_dict": F_f.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                'global_step': i,
                'psnrs': psnrs,
                'iternums': iternums,
            },
            path,
          )

print("Done!")