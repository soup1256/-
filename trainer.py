import os
import torch
from tqdm import tqdm
import utility

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.noise_g = args.noise_g
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.noise_level = args.initial_noise_level  # 初始化噪声水平

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'), map_location='cpu')
            )
            for _ in range(len(ckp.log)):
                self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr()[0]

        # 动态调整噪声水平
        self.noise_level = min(self.args.max_noise_level, self.noise_level + self.args.noise_increment_per_epoch)
        self.ckp.write_log(f'[Epoch {epoch}]\tLearning rate: {lr:.2e}\tNoise Level: {self.noise_level}')

        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.Timer(), utility.Timer()
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            lr, hr = self.prepare([lr, hr])
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, idx_scale=0)
            loss = self.loss(sr, hr)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            else:
                print(f'Skip this batch {batch + 1}! (Loss: {loss.item()})')

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log(f'[{(batch + 1) * self.args.batch_size}/{len(self.loader_train.dataset)}]\t'
                                   f'{self.loss.display_loss(batch)}\t'
                                   f'{timer_model.release():.1f}+{timer_data.release():.1f}s')

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.noise_g)))
        self.model.eval()

        timer_test = utility.Timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.noise_g):
                eval_acc = 0
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename, original_size) in enumerate(tqdm_test):  # 加入original_size
                    filename = filename[0]
                    lr, hr = self.prepare([lr, hr])

                    # 模型预测
                    sr = self.model(lr, idx_scale=0)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    # 应用细节增强
                    sr = utility.enhance_image(sr)

                    # 恢复到原始大小，确保 sr 是四维张量 (N, C, H, W)
                    if sr.dim() == 3:  # 如果是三维张量，添加批次维度
                        sr = sr.unsqueeze(0)

                    # 恢复尺寸
                    sr = torch.nn.functional.interpolate(sr, size=original_size[::-1], mode='bilinear', align_corners=False)  # 恢复到原始尺寸

                    # 如果需要，可以再次删除批次维度
                    if sr.size(0) == 1:
                        sr = sr.squeeze(0)

                    save_list = [sr]
                    eval_acc += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range
                    )
                    save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale, original_size)

                    torch.cuda.empty_cache()

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, l):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
