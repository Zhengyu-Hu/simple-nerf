# Simple-Nerf

## Train
! git clone https://github.com/Zhengyu-Hu/simple-nerf.git \
&& cd simple-nerf \
&& bash load_data.sh \
&& python train.py -n 20000 -l
## Inference
! git clone https://github.com/Zhengyu-Hu/simple-nerf.git \
&& cd simple-nerf \
&& bash data_for_test.sh && python inference.py

## install segment-anything-2
pip install 'git+https://github.com/facebookresearch/sam2.git'
