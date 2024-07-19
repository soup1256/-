import os
from data import common
import numpy as np
import imageio
import torch
import torch.utils.data as data

class Demo(data.Dataset):
    def __init__(self, args, train=False):
        self.args = args
        self.name = 'Demo'
        self.scale = args.scale
        self.idx_scale = 0
        self.train = train
        self.benchmark = False

        self.filelist = []
        for f in os.listdir(args.dir_demo):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.filelist.append(os.path.join(args.dir_demo, f))
        self.filelist.sort()

    def __getitem__(self, idx):
        filename = os.path.split(self.filelist[idx])[-1]
        filename, _ = os.path.splitext(filename)
        lr = imageio.imread(self.filelist[idx])
        lr = common.set_channel([lr], self.args.n_colors)[0]
        return common.np2Tensor([lr], self.args.rgb_range)[0], -1, filename

    def __len__(self):
        return len(self.filelist)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
