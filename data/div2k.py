import os
from data import common
from data import srdata
import numpy as np
import imageio
import torch
import torch.utils.data as data

class DIV2K(srdata.SRData):
    def __init__(self, args, train=True):
        super(DIV2K, self).__init__(args, train)
        self.repeat = args.test_every // (args.n_train // args.batch_size)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.noise_g]
        if self.train:
            idx_begin = 0
            idx_end = self.args.n_train
        else:
            idx_begin = self.args.n_train
            idx_end = self.args.offset_val + self.args.n_val

        for i in range(idx_begin + 1, idx_end + 1):
            filename = '{:0>4}'.format(i)
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            for si, s in enumerate(self.noise_g):
                list_lr[si].append(os.path.join(
                    self.dir_lr,
                    'X{}/{}_s{}{}'.format(s, filename, s, self.ext)
                ))

        list_hr.sort()
        for l in list_lr:
            l.sort()

        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'DIV2K_train')
        self.dir_hr = os.path.join(self.apath, 'Clean')
        self.dir_lr = os.path.join(self.apath, 'Noisy')
        self.ext = '.png'

    def _name_hrbin(self):
        return os.path.join(self.apath, 'bin', '{}_bin_HR.npy'.format(self.split))

    def _name_lrbin(self, scale):
        return os.path.join(self.apath, 'bin', '{}_bin_LR_X{}.npy'.format(self.split, scale))

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx
