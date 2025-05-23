import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from .sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
from collections import OrderedDict
from basicsr.utils.dist_util import master_only
import os
import os.path as osp
from basicsr.utils import get_root_logger, tensor2img, imwrite
from basicsr.metrics import calculate_metric
import pyiqa
import time
from tqdm import tqdm
import corun_colabator.archs.open_clip as open_clip

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


@MODEL_REGISTRY.register()
class Pretrain_by_Depth(SRModel):
    """
    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(Pretrain_by_Depth, self).__init__(opt)
        if self.is_train:
            self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
            if self.mixing_flag:
                mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
                use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
                self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        if self.is_train:
            if self.opt['clip_plugin'].get('use_clip', False) or self.opt['train'].get('use_clip_loss', False):
                self.init_clip()

    def init_clip(self):
        clip_model_type = self.opt['clip_plugin'].get('clip_model_type', None)
        checkpoint = self.opt['clip_plugin'].get('pretrained_clip_weight', None)
        tokenizer_type = self.opt['clip_plugin'].get('tokenizer_type', None)
        self.clip_better = self.opt['clip_plugin'].get('clip_better', None)
        self.degradation_type = self.opt['clip_plugin'].get('degradation_type', None)
        self.clip_model, self.clip_preprocess = open_clip.create_model_from_pretrained(clip_model_type,
                                                                                       pretrained=checkpoint)
        self.clip_model = self.model_to_device(self.clip_model)
        self.clip_model.eval()
        self.tokenizer = open_clip.get_tokenizer(tokenizer_type)
        degradations = ['motion-blurry', 'hazy', 'jpeg-compressed', 'low-light', 'noisy', 'raindrop', 'rainy',
                        'shadowed', 'snowy', 'uncompleted']
        text = self.tokenizer(degradations)
        text = text.to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            if self.opt['dist']:
                text_features = self.clip_model.module.encode_text(text)
            else:
                text_features = self.clip_model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            self.text_features = text_features


    def feed_data(self, data):
        try:
            del self.lq
            del self.gt
            del self.transmission
        except:
            pass

        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 't' in data:
            self.transmission = data['t'].to(self.device)

        if self.is_train and self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def pad_test(self, img, window_size):
        # scale = self.opt.get('scale', 1)
        scale = 1
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = img.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        lq = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return lq,mod_pad_h,mod_pad_w

    def test(self):
        window_size = self.opt['val'].get('window_size', 0)
        if window_size:
            lq, mod_pad_h, mod_pad_w = self.pad_test(self.lq,window_size)
        else:
            lq = self.lq

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.outputs, self.output_transmissions, self.step_images, self.recon_images = self.net_g_ema(
                    img=lq, debug=True)
                self.output = self.outputs[0].clamp(0,1)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.outputs, self.output_transmissions, self.step_images, self.recon_images = self.net_g(
                    img=lq, debug=True)
                self.output = self.outputs[0].clamp(0,1)
            self.net_g.train()

        if window_size:
            scale = self.opt.get('scale', 1)
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]


    def get_clip_hazy_rate(self, img):
        image = self.clip_preprocess(img)
        sum_probs = 0
        for degradation in self.degradation_type:
            with torch.no_grad(), torch.cuda.amp.autocast():
                if self.opt['dist']:
                    _, degra_features = self.clip_model.module.encode_image(image, control=True)
                else:
                    _, degra_features = self.clip_model.encode_image(image, control=True)
                # image_features /= image_features.norm(dim=-1, keepdim=True)
                degra_features /= degra_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * degra_features @ self.text_features.T).softmax(dim=-1)
                probs = text_probs[:, degradation]
                sum_probs = sum_probs + probs
        return sum_probs

    def get_batch_avg_hazy_rate(self, imgs):
        sum_rate = self.get_clip_hazy_rate(imgs)
        sum_rate = sum_rate.mean()
        return sum_rate / imgs.shape[0]


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.outputs, transmissions = self.net_g(self.lq, finetune=True)
        hazy_recon = self.outputs[0] * transmissions[0] + (1 - transmissions[0])

        l_total = 0
        loss_dict = OrderedDict()

        # pixel loss
        if self.cri_pix:

            l_pix = self.cri_pix(self.outputs[0], self.gt)
            loss_dict['l_pix'] = l_pix
            l_total += l_pix

            if self.opt["train"].get("asm_loss", True) is True:
                l_asm = self.cri_pix(hazy_recon, self.lq) * 0.01
                loss_dict['l_asm'] = l_asm
                l_total += l_asm

        if self.opt['train'].get('use_clip_loss', False):
            clip_loss = self.get_batch_avg_hazy_rate(self.outputs[0])
            loss_dict['clip_loss'] = clip_loss
            l_total += clip_loss

        # perceptual loss
        if self.cri_contrastperceptual:
            l_contrast_percep, l_contrast_style = self.cri_contrastperceptual(self.outputs[0], self.gt.detach(), self.lq.detach())
            if l_contrast_percep is not None:
                l_total += l_contrast_percep
                loss_dict['l_contrast_percep'] = l_contrast_percep
            if l_contrast_style is not None:
                l_total += l_contrast_style
                loss_dict['l_contrast_style'] = l_contrast_style


        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def save_all_stages_img(self, img_name, current_iter):
        if hasattr(self, 'gt'):
            output = torch.cat([self.lq, self.output, self.gt], dim=3)
        else:
            output = torch.cat([self.lq, self.output], dim=3)

        output_transmissions = self.output_transmissions[::-1]
        step_images = self.step_images[::-1]
        recon_images = self.recon_images[::-1]

        output_transmission = output_transmissions[0]
        step_image = step_images[0]
        recon_image = recon_images[0]

        for t in range(1, len(self.outputs)):
            step_image = torch.cat((step_image, step_images[t]), 3)
            output_transmission = torch.cat((output_transmission, output_transmissions[t]), 3)
            recon_image = torch.cat((recon_image, recon_images[t]), 3)

        # save image
        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                 f'{img_name}_{current_iter}_compare_img.png')
        save_trans_path = osp.join(self.opt['path']['visualization'], img_name,
                                   f'{img_name}_{current_iter}_step_trans.png')
        save_gt_path = osp.join(self.opt['path']['visualization'], img_name,
                                      f'{img_name}_{current_iter}_gt.png')
        save_recon_path = osp.join(self.opt['path']['visualization'], img_name,
                                        f'{img_name}_{current_iter}_step_recon.png')
        save_step_path = osp.join(self.opt['path']['visualization'], img_name,
                                        f'{img_name}_{current_iter}_step.png')


        output = output.detach().cpu()
        output_transmission = output_transmission.detach().cpu()
        step_img = step_image.detach().cpu()
        recon_img = recon_image.detach().cpu()

        if hasattr(self, 'gt'):
            gt_img = self.gt.detach().cpu()
            imwrite(tensor2img(gt_img), save_gt_path)
            del gt_img

        imwrite(tensor2img(output), save_img_path)
        imwrite(tensor2img(output_transmission),save_trans_path)
        imwrite(tensor2img(recon_img), save_recon_path)
        imwrite(tensor2img(step_img), save_step_path)

        del output
        del output_transmission
        del step_img
        del recon_img


    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        save_source_img = self.opt['val'].get('save_source', False)
        save_all_stages = self.opt['val'].get('save_all_stages', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        nima = pyiqa.create_metric('nima')
        brisque = pyiqa.create_metric('brisque')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            if save_all_stages:
                self.save_all_stages_img(img_name, current_iter)

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])

            metric_data['img'] = sr_img
            lq_img = tensor2img([visuals['lq']])
            metric_data['lq'] = lq_img

            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            metric_data['nima'] = nima
            metric_data['brisque'] = brisque

            # tentative for out of GPU memory
            del self.lq
            del self.output
            del self.outputs
            del self.output_transmissions
            del self.step_images
            del self.recon_images

            torch.cuda.empty_cache()

            if self.opt['is_train']:
                save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                if save_source_img:
                    save_lq_path = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_lq.png')
            else:
                save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')
                if save_source_img:
                        save_lq_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                f'{img_name}_lq.png')

            imwrite(sr_img, save_img_path)
            if save_source_img:
                imwrite(lq_img, save_lq_path)

            metric_data['img_path'] = save_img_path
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                metric_data['img2'] = gt_img


            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        del nima
        del brisque
        torch.cuda.empty_cache()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)


