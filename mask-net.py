import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import os
from helper import *
from load_data import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"==>> device: {device}")

class MaskNeRF(nn.Module):
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
            nn.Linear(net_width, 1), # 此处输出0-1掩码
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
        density = outputs[:, 0]
        color = self.late_mlp(torch.cat([outputs[:, 1:], ds_encoded], dim=-1))
        return {"color": color, "density": density}

mask_folder = "data/lego-mask"
images = load_images_from_folder(mask_folder)
splits = ["test"]
_, poses, K = get_data(splits)
focal = K[0,0]
poses = torch.tensor(poses.astype(np.float32))