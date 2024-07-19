import utility
from model import common
from loss import discriminator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Adversarial(nn.Module):
    def __init__(self, args, gan_type):
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = args.gan_k
        self.discriminator = discriminator.Discriminator(args, gan_type)
        if gan_type != 'WGAN_GP':
            self.optimizer = utility.make_optimizer(args, self.discriminator)
        else:
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=1e-5
            )
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.loss = 0

    def forward(self, fake, real):
        fake_detach = fake.detach()
        self.loss = 0

        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            d_fake = self.discriminator(fake_detach)
            d_real = self.discriminator(real)
            if self.gan_type == 'GAN':
                label_fake = torch.zeros_like(d_fake)
                label_real = torch.ones_like(d_real)
                loss_d = F.binary_cross_entropy_with_logits(d_fake, label_fake) + F.binary_cross_entropy_with_logits(d_real, label_real)
            elif 'WGAN' in self.gan_type:
                loss_d = (d_fake - d_real).mean()
                if 'GP' in self.gan_type:
                    epsilon = torch.rand_like(fake).view(-1, 1, 1, 1)
                    hat = fake_detach * (1 - epsilon) + real * epsilon
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * (gradient_norm - 1).pow(2).mean()
                    loss_d += gradient_penalty

            self.loss += loss_d.item()
            loss_d.backward()
            self.optimizer.step()

            if self.gan_type == 'WGAN':
                for p in self.discriminator.parameters():
                    p.data.clamp_(-1, 1)

        self.loss /= self.gan_k

        d_fake_for_g = self.discriminator(fake)
        if self.gan_type == 'GAN':
            label_real = torch.ones_like(d_fake_for_g)
            loss_g = F.binary_cross_entropy_with_logits(d_fake_for_g, label_real)
        elif 'WGAN' in self.gan_type:
            loss_g = -d_fake_for_g.mean()

        return loss_g

    def state_dict(self, *args, **kwargs):
        state_discriminator = self.discriminator.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()
        return dict(discriminator=state_discriminator, optimizer=state_optimizer)

    def load_state_dict(self, state_dict, strict=True):
        self.discriminator.load_state_dict(state_dict['discriminator'], strict)
        self.optimizer.load_state_dict(state_dict['optimizer'])
