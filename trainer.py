import os
import torch
from tqdm import tqdm
import utility
import piq

import easyocr

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
        self.noise_level = args.initial_noise_level

        # Initialize OCR Reader
        self.ocr_reader = easyocr.Reader(['en'])  # 可以根据需要选择不同语言

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'), map_location='cpu')
            )
            for _ in range(len(ckp.log)):
                self.scheduler.step()

        self.error_last = 1e8

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.noise_g)))
        self.model.eval()

        timer_test = utility.Timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.noise_g):
                eval_psnr = 0
                eval_ssim = 0
                eval_vif = 0
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename, original_size) in enumerate(tqdm_test):
                    filename = filename[0]
                    lr, hr = self.prepare([lr, hr])

                    # 模型预测
                    sr = self.model(lr, idx_scale=0)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    # 应用细节增强
                    sr = utility.enhance_image(sr)

                    # 恢复到原始大小
                    if sr.dim() == 3:
                        sr = sr.unsqueeze(0)
                    sr = torch.nn.functional.interpolate(sr, size=original_size[::-1], mode='bilinear', align_corners=False)

                    if sr.size(0) == 1:
                        sr = sr.squeeze(0)

                    # 确保输入到 piq 函数的张量是 4 维的
                    sr = sr.unsqueeze(0) if sr.dim() == 3 else sr
                    hr = hr.unsqueeze(0) if hr.dim() == 3 else hr

                    # 计算 PSNR、SSIM 和 VIF
                    psnr_value = utility.calc_psnr(sr, hr, scale, self.args.rgb_range)
                    ssim_value = piq.ssim(sr, hr, data_range=self.args.rgb_range).item()
                    vif_value = piq.vif_p(sr, hr, data_range=self.args.rgb_range).item()

                    eval_psnr += psnr_value
                    eval_ssim += ssim_value
                    eval_vif += vif_value
                    # 在去噪后的图像上执行 OCR
                    ocr_results = self.perform_ocr_on_image(sr)

                    # 打印 OCR 结果
                    for result in ocr_results:
                        detected_text, confidence = result[1], result[2]
                        # 只记录 OCR 日志，不在终端显示
                        self.ckp.write_ocr_log(f'OCR Detected text: "{detected_text}", Confidence: {confidence:.2f}')

                    save_list = [sr]
                    save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale, original_size)

                    torch.cuda.empty_cache()

                # 计算平均值
                avg_psnr = eval_psnr / len(self.loader_test)
                avg_ssim = eval_ssim / len(self.loader_test)
                avg_vif = eval_vif / len(self.loader_test)

                self.ckp.log[-1, idx_scale] = avg_psnr
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {}), SSIM: {:.3f}, VIF: {:.3f}'.format(
                        self.args.data_test,
                        scale,
                        avg_psnr,
                        best[0][idx_scale],
                        best[1][idx_scale] + 1,
                        avg_ssim,
                        avg_vif
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def perform_ocr_on_image(self, image_tensor):
        """
        使用 EasyOCR 在给定的图像上执行 OCR 识别。

        :param image_tensor: PyTorch Tensor, 经过去噪的图像
        :return: OCR 识别结果
        """
        # 如果图像是 4 维张量 [B, C, H, W]，取第一个样本
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]

        # 确保图像为 3 维张量 [C, H, W]
        if image_tensor.dim() == 3:
            # 将 PyTorch 张量转换为 NumPy 数组并还原为图像
            image = image_tensor.mul(255).byte().cpu().numpy().transpose(1, 2, 0)
        else:
            raise ValueError("Expected 3D tensor with shape [C, H, W], got {}D tensor".format(image_tensor.dim()))

        # 执行 OCR 识别
        results = self.ocr_reader.readtext(image)
        return results


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
