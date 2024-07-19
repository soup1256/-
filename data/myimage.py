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

        self.clean_dir = os.path.abspath(os.path.join(args.testpath, 'clean'))
        self.noisy_dir = os.path.abspath(os.path.join(args.testpath, 'noisy'))

        if not os.path.exists(self.clean_dir):
            raise FileNotFoundError(f'Clean directory not found: {self.clean_dir}')
        if not os.path.exists(self.noisy_dir):
            raise FileNotFoundError(f'Noisy directory not found: {self.noisy_dir}')

        self.clean_files = sorted([os.path.join(self.clean_dir, f) for f in os.listdir(self.clean_dir) if f.lower().endswith('.bmp')])
        self.noisy_files = sorted([os.path.join(self.noisy_dir, f) for f in os.listdir(self.noisy_dir) if f.lower().endswith('.bmp')])

    def __getitem__(self, idx):
        filename = os.path.split(self.noisy_files[idx])[-1]
        filename, _ = os.path.splitext(filename)

        noisy = imageio.imread(self.noisy_files[idx])
        clean = imageio.imread(self.clean_files[idx])

        # Convert 4-channel image to 3-channel if necessary
        if noisy.shape[-1] == 4:
            noisy = noisy[:, :, :3]
        if clean.shape[-1] == 4:
            clean = clean[:, :, :3]

        # Resize images while maintaining the aspect ratio if they are too large
        max_dim = 512
        if max(noisy.shape[:2]) > max_dim:
            ratio = max_dim / max(noisy.shape[:2])
            noisy = resize(noisy, (int(noisy.shape[0] * ratio), int(noisy.shape[1] * ratio)), preserve_range=True, anti_aliasing=True).astype(noisy.dtype)
        if max(clean.shape[:2]) > max_dim:
            ratio = max_dim / max(clean.shape[:2])
            clean = resize(clean, (int(clean.shape[0] * ratio), int(clean.shape[1] * ratio)), preserve_range=True, anti_aliasing=True).astype(clean.dtype)

        noisy = common.set_channel([noisy], self.args.n_colors)[0]
        clean = common.set_channel([clean], self.args.n_colors)[0]

        noisy, clean = common.np2Tensor([noisy, clean], self.args.rgb_range)

        return noisy, clean, filename

    def __len__(self):
        return len(self.noisy_files)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
