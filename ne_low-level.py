"Author: Zhisen Cong"

from scipy.io import loadmat
import math
from layers.transformer import Transformer
import argparse


# ne_freq
sequence_length = 10
# can change
patch_len = 10
# can change
# calculate number of patches
n_patches = 1

print("Number of patches:", n_patches)

ne_transformer = Transformer(
    patch_len=patch_len,
    n_patches=n_patches,
    e_layers=2,
    c_in=1,
    # important for concat with eeg and emg data
    inner_dim=128,
    n_heads=4,
    d_head=16,
    dropout=0.1,
    path_drop=0.0,
    activation='relu',
    norm='layernorm',
    mult=4,
    mix_type=0,
    cls=True,
    flag='seq',
    domain='time',
    output_attentions=False,
)


matfile = loadmat("groundtruth_data/arch_387.mat")
ne = matfile['ne'].flatten()
ne_freq = matfile['ne_frequency'].item()
print("len(ne) is: ",len(ne))
print("ne_freq is: ",math.floor(ne_freq))
n_seconds = 64
print("n_seconds is: ",n_seconds)
shortenedNE = ne[:n_seconds*math.floor(ne_freq)]

import torch
ne_tensor = torch.tensor(shortenedNE, dtype=torch.float32)
reshapedNE = ne_tensor.reshape(1,n_seconds,1,int(ne_freq))
print(reshapedNE.shape)


ne_output,_ = ne_transformer(reshapedNE)
print("ne_output.shape is: ",ne_output.shape) # shape is [1, n_seconds, n_patches+1, inner_dim]
#print("ne_output is",ne_output)
print(ne_output[:, :, -1].shape)

