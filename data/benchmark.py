import os
from data import common
from data import srdata
import numpy as np
import imageio
import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, train=True):
        super(Benchmark, self).__init__(args, train, benchmark=True)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.noise_g]
        for entry in os.scandir(self.dir_hr):
            if entry.is_file():
                filename = os.path.splitext(entry.name)[0]
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
        self.apath = os.path.join(dir_data, 'benchmark', self.args.data_test)
        self.dir_hr = os.path.join(self.apath, 'Clean')
        self.dir_lr = os.path.join(self.apath, 'Noisy')
        self.ext = '.png'
