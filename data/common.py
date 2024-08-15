
import random
import numpy as np
import imageio
import skimage.color as sc
import torch

def get_patch(img_in, img_tar, patch_size, scale, multi_scale=False):
    ih, iw = img_in.shape[:2]
    p = scale if multi_scale else 1
    ip = p * patch_size
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[iy:iy + ip, ix:ix + ip, :]

    return img_in, img_tar

def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]

def _np2Tensor(img, rgb_range):
    # 将 PIL.Image 对象转换为 numpy 数组
    if isinstance(img, torch.Tensor):
        np_transpose = img.numpy().transpose((2, 0, 1))
    else:
        np_transpose = np.array(img).transpose((2, 0, 1))

    tensor = torch.from_numpy(np_transpose).float()
    tensor.mul_(rgb_range / 255)
    return tensor

def np2Tensor(l, rgb_range):
    return [_np2Tensor(_l, rgb_range) for _l in l]

def add_noise(x, noise_type='G', noise_value=25):
    if noise_type == 'G':  # 高斯噪声
        noises = np.random.normal(scale=noise_value, size=x.shape)
        noises = noises.round()
    elif noise_type == 'S':  # 泊松噪声
        noises = np.random.poisson(x * noise_value) / noise_value
        noises = noises - noises.mean(axis=0).mean(axis=0)
    elif noise_type == 'SP':  # 盐和胡椒噪声
        prob = noise_value / 255.0
        noise = np.random.rand(*x.shape)
        x[noise < prob / 2] = 0
        x[noise > 1 - prob / 2] = 255
        noises = np.zeros_like(x)
    elif noise_type == 'speckle':  # 斑点噪声
        noises = np.random.randn(*x.shape) * noise_value
        noises = noises.round()
    else:
        return x

    x_noise = x.astype(np.int16) + noises.astype(np.int16)
    x_noise = x_noise.clip(0, 255).astype(np.uint8)
    return x_noise

def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img

    return [_augment(_l) for _l in l]

def add_augmentation(image):
    # 添加随机噪声
    noise_types = ['G', 'S', 'SP', 'speckle']
    noise_type = random.choice(noise_types)
    noise_value = random.randint(10, 50)
    image = add_noise(image, noise_type=noise_type, noise_value=noise_value)
    
    return image
