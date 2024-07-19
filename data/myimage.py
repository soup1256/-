import os
from data import common
import numpy as np
import torch
import torch.utils.data as data
import imageio
from skimage.transform import resize

class MyImage(data.Dataset):
    def __init__(self, args, train=False):
        self.args = args
        self.name = 'MyImage'
        self.scale = args.noise_g
        self.idx_scale = 0
        self.train = train
        self.benchmark = False

        self.filelist_lr = []

        lr_dir = os.path.join(args.testpath, args.testset, 'X1')

        for f in os.listdir(lr_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.filelist_lr.append(os.path.join(lr_dir, f))

        self.filelist_lr.sort()

    def __getitem__(self, idx):
        filename = os.path.split(self.filelist_lr[idx])[-1]
        filename, _ = os.path.splitext(filename)

        lr = imageio.imread(self.filelist_lr[idx])

        # Convert 4-channel image to 3-channel if necessary
        if lr.shape[-1] == 4:
            lr = lr[:, :, :3]

        lr = common.set_channel([lr], self.args.n_colors)[0]

        return common.np2Tensor([lr], self.args.rgb_range)[0], -1, filename

    def __len__(self):
        return len(self.filelist_lr)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
