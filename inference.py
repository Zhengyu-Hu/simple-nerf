import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import os
from helper import *
from tqdm import trange
from load_data import *

# load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
F_c = NeRF().to(device)
F_f = NeRF().to(device)
model_path = './model'
# Number of query points passed through the MLP at a time.
chunk_size = 1024 * 32
# ckpt
LOAD_ckpts = True
if LOAD_ckpts:
  ckpts = [os.path.join(model_path,f) for f in sorted(os.listdir(model_path)) if 'tar' in f]
  # 升序排列，选择最后一轮开始训练
  if len(ckpts)>0:
    print('Found ckpts',f'-->{ckpts}')
    checkpoint = torch.load(ckpts[-1], weights_only=False)
    step = checkpoint['global_step']
    F_c.load_state_dict(checkpoint['model_coarse_state_dict'])
    F_f.load_state_dict(checkpoint['model_fine_state_dict'])
  else:
    print('No ckpts found')
    # print('Train from scratch')

# load data
# 导入数据
splits = ['test']
imgs, poses, K = get_data(splits)
images = imgs[...,:3]*imgs[...,-1:] + (1.-imgs[...,-1:])
focal = K[0,0]
images = torch.tensor(images.astype(np.float32))
poses = torch.tensor(poses.astype(np.float32))
# img_size = images.shape[1]
H, W = images.shape[1:3]

# rendering args
# Near bound. See Section 4.
t_n = 2.0
# Far bound. See Section 4.
t_f = 6.0
# Number of coarse samples along a ray. See Section 5.3.
N_c = 64
# Number of fine samples along a ray. See Section 5.3.
N_f = 128
# Bins used to sample depths along a ray. See Equation (2) in Section 4.
t_i_c_gap = (t_f - t_n) / N_c
t_i_c_bin_edges = (t_n + torch.arange(N_c) * t_i_c_gap).to(device)

inference_path = './inference'
if not os.path.exists(inference_path):
   os.mkdir(inference_path)
def run_one_inference(img, pose, idx):
    pose = pose.to(device)
    rays_o, rays_d = get_rays(H, W, focal, pose)
    F_c.eval()
    F_f.eval()
    with torch.no_grad():
        (_, C_rs_f) = run_one_iter_of_nerf(
            rays_o,
            rays_d,
            t_i_c_bin_edges,
            t_i_c_gap,
            N_c,
            N_f,
            chunk_size,
            F_c,
            F_f,
        )
    target_fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(C_rs_f.detach().cpu().numpy())
    plt.subplot(122)
    plt.imshow(img)
    target_fig.savefig(inference_path + f'/{step}-pose{idx}.png')
    plt.close()

N = images.shape[0]
for idx in trange(N):
    # idx = 0
    run_one_inference(images[idx], poses[idx], idx)