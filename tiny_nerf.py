import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import os
from helper import *
from load_data import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"==>> device: {device}")


class TinyNeRF(nn.Module):
    def __init__(self):
        super().__init__()
        # 0 means no position encoding
        self.L_pos = 6
        self.L_dir = 4
        pos_enc_feats = 3 + 3 * 2 * self.L_pos
        dir_enc_feats = 3 + 3 * 2 * self.L_dir

        # 线性层宽改为128
        net_width = 256
        self.early_mlp = nn.Sequential(
            nn.Linear(pos_enc_feats, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width + 1), # 此处输出密度值
            nn.ReLU(),
        )
        self.late_mlp = nn.Sequential(
            nn.Linear(net_width + dir_enc_feats, net_width),
            nn.ReLU(),
            nn.Linear(net_width, 3), # 此处输出RGB
            nn.Sigmoid(),
        )

    def forward(self, xs, ds):
        xs_encoded = [xs]
        for l_pos in range(self.L_pos):
            xs_encoded.append(torch.sin(2 ** l_pos * torch.pi * xs))
            xs_encoded.append(torch.cos(2 ** l_pos * torch.pi * xs))

        xs_encoded = torch.cat(xs_encoded, dim=-1)

        ds = ds / ds.norm(p=2, dim=-1).unsqueeze(-1)
        ds_encoded = [ds]
        for l_dir in range(self.L_dir):
            ds_encoded.append(torch.sin(2 ** l_dir * torch.pi * ds))
            ds_encoded.append(torch.cos(2 ** l_dir * torch.pi * ds))

        ds_encoded = torch.cat(ds_encoded, dim=-1)

        outputs = self.early_mlp(xs_encoded)
        sigma_is = outputs[:, 0]
        c_is = self.late_mlp(torch.cat([outputs[:, 1:], ds_encoded], dim=-1))
        return {"color": c_is, "density": sigma_is}    

def run_one_iter_of_tinynerf(rays_o, rays_d, t_edges, t_gap, N_c, chunk_size, F):
    r, t = get_coarse_query_pts(rays_o, rays_d, N_c, t_edges, t_gap)
    C, w = render(r, t, rays_d, chunk_size, F)
    return C


data_f = "car.npz"
data = np.load(data_f)
images = torch.tensor(data["images"].astype(np.float32))
if data_f == "car.npz":
    images = images/255
poses = torch.tensor(data["poses"].astype(np.float32))
focal = float(data['focal'])
H, W = images.shape[1:3]


test_idx = 150
test_fig = plt.figure()
plt.imshow(images[test_idx])
test_fig.savefig(f'./logs/test_img_{test_idx}.jpg')
test_img = torch.Tensor(images[test_idx]).to(device)
test_os, test_ds = get_rays(H, W, focal, poses[test_idx])
test_os = test_os.to(device) # [H, W, 3]
test_ds = test_ds.to(device)

seed = 9458
torch.manual_seed(seed)
np.random.seed(seed)

F = TinyNeRF().to(device)
chunk_size = 16384

lr = 5e-3
optimizer = torch.optim.Adam(F.parameters(), lr)
criterion = nn.MSELoss()

t_n = 1.0
t_f = 4.0
N_c = 32
t_gap = (t_f - t_n) / N_c
t_edges = (t_n + torch.arange(N_c) * t_gap).to(device)

train_idxs = np.arange(len(images)) != test_idx
images = torch.Tensor(images[train_idxs])
poses = torch.Tensor(poses[train_idxs])
psnrs = []
iternums = []
num_iters = 20000
display_every = 100

F.train()
for i in range(num_iters):
    optimizer.zero_grad()

    target_img_idx = np.random.randint(images.shape[0])
    target_img = images[target_img_idx].to(device)
    target_pose = poses[target_img_idx].to(device)
    rays_o, rays_d = get_rays(H, W, focal, target_pose)

    C = run_one_iter_of_tinynerf(rays_o, rays_d, t_edges, t_gap, N_c, chunk_size, F)
    loss = criterion(C, target_img)
    
    loss.backward()
    optimizer.step()

    if i % display_every == 0 :
        F.eval()
        with torch.no_grad():
            C = run_one_iter_of_tinynerf(test_os, test_ds, t_edges, t_gap, N_c, chunk_size, F)

        loss = criterion(C, test_img)
        print(f"Loss: {loss.item()}")
        psnr = -10.0 * torch.log10(loss)

        psnrs.append(psnr.item())
        iternums.append(i)

        eval_fig = plt.figure(figsize=(10, 4))
        plt.subplot(121)
        plt.imshow(C.detach().cpu().numpy())
        plt.title(f"Iteration {i}")
        plt.subplot(122)
        plt.plot(iternums, psnrs)
        plt.title("PSNR")
        eval_fig.savefig(f'./logs/iter_{i}.jpg')

        F.train()