import os
import math
import time
import datetime
import numpy as np
import imageio
import torch
import cv2


import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

class Timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0
        return ret

    def reset(self):
        self.acc = 0

class Checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if args.load == '.':
            if args.save == '.': args.save = now
            self.dir = '../experiment/' + args.save
        else:
            self.dir = '../experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.log)))

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = '.'

        self._make_dir(self.dir)
        self._make_dir(self.dir + '/model')
        self._make_dir(self.dir + '/results')

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def save(self, trainer, epoch, is_best=False):
        target = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        target.save(self.dir, epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        torch.save(self.log, os.path.join(self.dir, 'psnr_log.pt'))
        torch.save(trainer.optimizer.state_dict(), os.path.join(self.dir, 'optimizer.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, len(self.log))
        label = 'Denoise on {}'.format(self.args.data_test)
        fig = plt.figure()
        plt.title(label)
        for idx_scale, noise in enumerate(self.args.noise_g):
            if len(self.log[:, idx_scale]) == len(axis):
                plt.plot(axis, self.log[:, idx_scale].numpy(), label='Noise {}'.format(noise))
            else:
                print(f"Skipping scale {idx_scale} due to dimension mismatch: axis length={len(axis)}, log length={len(self.log[:, idx_scale])}")

        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig('{}/test_{}.pdf'.format(self.dir, self.args.data_test))
        plt.close(fig)

    def save_results(self, filename, save_list, scale, original_size):
        filename = '{}/results/{}_x{}_SR.png'.format(self.dir, filename, scale)
        sr = save_list[0]

        # 确保 sr 是 PyTorch 张量
        if isinstance(sr, torch.Tensor):
            # 如果 sr 是 4 维 (N, C, H, W) 则取第一个样本
            if sr.dim() == 4:
                sr = sr[0]

            # 恢复到原始大小
            sr = torch.nn.functional.interpolate(sr.unsqueeze(0), size=original_size[::-1], mode='bilinear', align_corners=False).squeeze(0)

            # 将值从 [0, 1] 恢复到 [0, 255] 范围内
            normalized = sr.data.mul(255).clamp(0, 255).byte()
            
            # 转换为 NumPy 数组并调换维度以匹配 (H, W, C)
            ndarr = normalized.permute(1, 2, 0).cpu().numpy()

            # Debugging 信息
            print(f"Saving SR: min={ndarr.min()}, max={ndarr.max()}")

            # 保存图像
            imageio.imwrite(filename, ndarr)
        else:
            raise ValueError(f"Expected PyTorch tensor, but got {type(sr)} instead.")



    
    def _make_dir(self, path):
        if not os.path.exists(path): 
            os.makedirs(path)


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    img = img.mul(pixel_range).clamp(0, 255).round()
    return img.div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    diff = (sr - hr).data.div(rgb_range)
    shave = scale
    if diff.size(1) > 1:
        convert = diff.new(1, 3, 1, 1)
        convert[0, 0, 0, 0] = 65.738
        convert[0, 1, 0, 0] = 129.057
        convert[0, 2, 0, 0] = 25.064
        diff.mul_(convert).div_(256)
        diff = diff.sum(dim=1, keepdim=True)

    mse = diff.pow(2).mean()

    return -10 * math.log10(mse)

def make_optimizer(args, my_model):
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())
    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {'betas': (args.beta1, args.beta2), 'eps': args.epsilon}
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = args.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)

def make_scheduler(args, my_optimizer):
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(my_optimizer, step_size=args.lr_decay, gamma=args.gamma)
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(int, milestones))
        scheduler = lrs.MultiStepLR(my_optimizer, milestones=milestones, gamma=args.gamma)

    return scheduler


def enhance_image(image):
    if image.dim() == 2:
        image = image.unsqueeze(0)
    elif image.dim() == 4:
        image = image[0]

    image_np = image.cpu().numpy().transpose(1, 2, 0)
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

    # 仅进行基本的亮度调整和去噪，避免任何对比度和颜色的增强
    image_np = cv2.fastNlMeansDenoisingColored(image_np, None, 10, 10, 7, 21)

    return torch.from_numpy(image_np).permute(2, 0, 1).float().div(255).to('cuda')
