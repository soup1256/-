from PIL import Image
import os
from data import common
import torch
import torch.utils.data as data
from torchvision import transforms

class MyImage(data.Dataset):
    def __init__(self, args, train=True, start_noise_level=10, noise_increment=5):
        self.args = args
        self.train = train
        self.current_noise_level = start_noise_level
        self.noise_increment = noise_increment
        
        if train:
            self.clean_dir = os.path.abspath(os.path.join(args.dir_data, 'train_cleaned'))
            self.noisy_dir = os.path.abspath(os.path.join(args.dir_data, 'train'))
        else:
            self.clean_dir = os.path.abspath(os.path.join(args.testpath, 'clean'))
            self.noisy_dir = os.path.abspath(os.path.join(args.testpath, 'noisy'))

        self.clean_images = sorted(os.listdir(self.clean_dir))
        self.noisy_images = sorted(os.listdir(self.noisy_dir))

        assert len(self.clean_images) == len(self.noisy_images), "清晰图像和带噪声图像数量不匹配"

        # 这里移除了Resize操作
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        noisy_image_path = os.path.join(self.noisy_dir, self.noisy_images[idx])
        clean_image_path = os.path.join(self.clean_dir, self.clean_images[idx])

        noisy_image = Image.open(noisy_image_path).convert('RGB')
        clean_image = Image.open(clean_image_path).convert('RGB')

        original_size = clean_image.size  # 获取图像的原始尺寸

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        noisy_image = self.add_progressive_noise(noisy_image)

        return noisy_image, clean_image, self.noisy_images[idx], original_size  # 返回原始尺寸

    def __len__(self):
        return len(self.noisy_images)

    def add_progressive_noise(self, image):
        noise = torch.randn_like(image) * (self.current_noise_level / 255.0)
        noisy_image = image + noise
        return torch.clamp(noisy_image, 0, 1)

    def increase_noise_level(self):
        self.current_noise_level += self.noise_increment
