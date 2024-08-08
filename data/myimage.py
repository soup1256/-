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

        self.image_dir = os.path.abspath(args.testpath)
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f'Image directory not found: {self.image_dir}')

        self.image_files = sorted([os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

        print(f"Found {len(self.image_files)} image files in {self.image_dir}")

    def __getitem__(self, idx):
        filename = os.path.split(self.image_files[idx])[-1]
        filename, _ = os.path.splitext(filename)

        image = imageio.imread(self.image_files[idx])

        # Convert 4-channel image to 3-channel if necessary
        if image.shape[-1] == 4:
            image = image[:, :, :3]

        # Resize images while maintaining the aspect ratio if they are too large
        max_dim = 512
        if max(image.shape[:2]) > max_dim:
            ratio = max_dim / max(image.shape[:2])
            image = resize(image, (int(image.shape[0] * ratio), int(image.shape[1] * ratio)), preserve_range=True, anti_aliasing=True).astype(image.dtype)

        image = common.set_channel([image], self.args.n_colors)[0]
        image_tensor = common.np2Tensor([image], self.args.rgb_range)[0]

        return image_tensor, -1, filename

    def __len__(self):
        return len(self.image_files)

    def set_scale(self, idx_scale):
        self.idx_scale = idx_scale
