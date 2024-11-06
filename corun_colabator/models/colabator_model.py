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
import corun_colabator.archs.memory_bank as memory_bank
import torchvision.transforms as TF
from torch.distributed.algorithms.join import Join

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
class Colabator(SRModel):
    """
    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(Colabator, self).__init__(opt)
        if self.is_train:
            self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
            if self.mixing_flag:
                mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
                use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
                self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        if self.is_train:
            if self.opt['colabator'].get('use_clip', False) or self.opt['train'].get('use_clip_loss', False):
                self.init_clip()
            if self.opt['colabator'].get('use_nr_iqa', False):
                self.init_nriqa()
            self.init_mmb()

    def init_clip(self):
        clip_model_type = self.opt['colabator'].get('clip_model_type', None)
        checkpoint = self.opt['colabator'].get('pretrained_clip_weight', None)
        tokenizer_type = self.opt['colabator'].get('tokenizer_type', None)
        self.clip_better = self.opt['colabator'].get('clip_better', None)
        self.degradation_type = self.opt['colabator'].get('degradation_type', None)
        self.block_size = self.opt['colabator'].get('block_size', None)
        self.weight_map_calculation = self.opt['colabator'].get('weight_map_calculation', 'addition')

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

    def init_nriqa(self):
        nr_iqa_type = self.opt['colabator'].get('nr_iqa_type', None)
        self.nr_iqa_better = self.opt['colabator'].get('nr_iqa_better', None)
        self.nr_iqa_scale = self.opt['colabator'].get('nr_iqa_scale', None)
        self.nr_iqa = pyiqa.create_metric(nr_iqa_type)
        self.nr_iqa = self.model_to_device(self.nr_iqa).eval()

    def init_mmb(self):
        self.memory_bank = memory_bank.Memory_bank_woT().to('cpu')
        memory_bank_path = self.opt['path'].get('pretrain_network_memory_bank', None)
        if memory_bank_path is not None:
            self.memory_bank.load_state_dict(torch.load(memory_bank_path))

    def block_image(self, image, block_size):
        B, C, H, W = image.size()
        BH, BW = block_size

        # Calculate the image shape after the block
        num_H = H // BH
        num_W = W // BW

        # Reshape the image into a block shape
        blocked_image = image.view(B, C, num_H, BH, num_W, BW)

        # Exchange dimensions so that the blocks are in the right place.
        blocked_image = blocked_image.permute(2, 4, 0, 1, 3, 5).contiguous()

        # Reshaped to the original shape
        blocked_image = blocked_image.view(num_H * num_W * B, C, BH, BW)

        return blocked_image

    def unblock_image(self, blocked_image, block_size, original_shape):
        B, C, H, W = original_shape
        BH, BW = block_size

        # Calculate the image shape after the block
        num_H = H // BH
        num_W = W // BW

        # Reshape the image into a block shape
        blocked_image = blocked_image.view(num_H, num_W, B, 1, 1)

        # Exchange dimensions so that the blocks are in the right place.
        blocked_image = blocked_image.permute(2, 0, 3, 1, 4).contiguous()

        # Reshaped to the original shape
        blocked_image = blocked_image.view(B, 1, num_H, num_W)

        # Resize to original shape
        blocked_image = torch.nn.functional.interpolate(blocked_image, (H, W), mode='bilinear', align_corners=False)

        return blocked_image

    def get_clip_degrad_rate(self, img):
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
                probs = text_probs[:,degradation]
                sum_probs = sum_probs + probs
        return sum_probs

    def get_batch_avg_degrad_rate(self, imgs):
        sum_rate = self.get_clip_degrad_rate(imgs)
        sum_rate = sum_rate.mean()
        return sum_rate / imgs.shape[0]

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'real' in data:
            self.real = data['real'].to(self.device)
        if 't' in data:
            self.transmission = data['t'].to(self.device)
        if 'real_strong' in data:
            self.real_strong = data['real_strong'].to(self.device)
        if 'real_name' in data:
            self.real_name = data['real_name']
        if 'mini_gt_size' in data:
            self.mini_gt_size = data['mini_gt_size']
        if 'gt_size' in data:
            self.gt_size = data['gt_size']

        if self.is_train and self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)


    def test(self):
        window_size = self.opt['val'].get('window_size', 0)
        if window_size:
            lq, mod_pad_h, mod_pad_w = self.pad_test(self.lq,window_size)

        else:
            lq = self.lq

        # if hasattr(self, 'net_g_ema'):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.outputs = self.net_g_ema(lq)
                self.output = self.outputs[0].clamp(0,1)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.outputs = self.net_g(lq)
                self.output = self.outputs[0].clamp(0,1)
            self.net_g.train()

        if window_size:
            scale = self.opt.get('scale', 1)
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def labal_selection(self, teacher): # teacher is pseudo label
        teacher_tar = teacher.detach()
        original_shape = teacher_tar.size()
        with torch.no_grad():
            # block image
            teacher_tar_blocks = self.block_image(teacher_tar, (self.block_size, self.block_size))
            if self.opt['colabator'].get('use_nr_iqa', False):
                # local
                teacher_nr_iqa_score_sequence = self.nr_iqa(teacher_tar_blocks)
                # global
                teacher_nr_iqa_score = (self.nr_iqa(teacher_tar) - self.nr_iqa_scale[0]) / (self.nr_iqa_scale[1] - self.nr_iqa_scale[0])
                # unblock image
                if self.nr_iqa_scale != 'sigmoid':
                    teacher_nr_iqa_score_mask = (self.unblock_image(teacher_nr_iqa_score_sequence,
                                                               (self.block_size, self.block_size), original_shape) - self.nr_iqa_scale[0]) / (self.nr_iqa_scale[1] - self.nr_iqa_scale[0])
                else:
                    teacher_nr_iqa_score_mask = torch.sigmoid(self.unblock_image(teacher_nr_iqa_score_sequence,
                                                               (self.block_size, self.block_size), original_shape))
                if self.nr_iqa_better == 'higher':
                    teacher_nr_iqa_score_mask = teacher_nr_iqa_score_mask
                    teacher_nr_iqa_score = teacher_nr_iqa_score
                else:
                    teacher_nr_iqa_score_mask = 1 - teacher_nr_iqa_score_mask
                    teacher_nr_iqa_score = 1 - teacher_nr_iqa_score
            else:
                teacher_nr_iqa_score_mask = 0
                teacher_nr_iqa_score = 0

            if self.opt['colabator'].get('use_clip', False):
                # local
                teacher_score_sequence = self.get_clip_degrad_rate(teacher_tar_blocks)
                # global
                teacher_score = self.get_clip_degrad_rate(teacher_tar)
                # unblock image
                teacher_score_mask = len(self.degradation_type) - self.unblock_image(teacher_score_sequence, (self.block_size, self.block_size),
                                                            original_shape)
                if self.clip_better == 'higher':
                    teacher_score = teacher_score
                    teacher_score_mask = teacher_score_mask
                else:
                    teacher_score = len(self.degradation_type) - teacher_score
                    teacher_score_mask = len(self.degradation_type) - teacher_score_mask
            else:
                teacher_score_mask = 0
                teacher_score = 0

        # final mask
        if self.weight_map_calculation == 'multiplication':
            teacher_mask = teacher_nr_iqa_score_mask * (teacher_score_mask / len(self.degradation_type))
        else:
            teacher_mask = (teacher_nr_iqa_score_mask + teacher_score_mask) / (len(self.degradation_type) + 1)

        teacher, teacher_nr_iqa_score, teacher_score, teacher_mask = self.memory_bank(self.real_name, teacher, teacher_nr_iqa_score, teacher_score, self.device, teacher_mask)
        teacher = teacher.to(self.device)

        return teacher, teacher_mask


    def optimize_parameters(self, current_iter, log_vars=None):
        if current_iter % self.opt['logger']['save_checkpoint_freq'] == 0:
            self.save_memory_bank(current_iter)

        self.optimizer_g.zero_grad()
        outputs = self.net_g(self.lq)
        output = outputs[0].clamp(0, 1)

        with torch.no_grad():
            pseudo_label = self.net_g_ema(self.real)

        pseudo_label = pseudo_label[0].clamp(0, 1)

        real_outputs = self.net_g(self.real_strong)
        real_output = real_outputs[0]
        pseudo_label, pseudo_mask = self.labal_selection(pseudo_label)

        ###########################################
        # this is a sample
        # you can modify this part for your losses

        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_pix:
            l_pix = self.cri_pix(output, self.gt).mean()
            if pseudo_label is not None:
                l_pix += (self.cri_pix(real_output, pseudo_label) * pseudo_mask * 2).mean()
            loss_dict['l_pix'] = l_pix
            l_pix = l_pix * 5
            l_total += l_pix

        if self.opt['train'].get('use_clip_loss', False):
            clip_loss = self.get_batch_avg_degrad_rate(output)
            clip_loss += self.get_batch_avg_degrad_rate(real_output)
            loss_dict['clip_loss'] = clip_loss
            l_total += clip_loss

        # perceptual loss
        if self.cri_contrastperceptual:
            l_percep, l_style = self.cri_contrastperceptual.standard_perceptual_loss(output, self.gt)
            l_contrast_percep, l_contrast_style = self.cri_contrastperceptual(real_output, pseudo_label, self.real_strong)
            if l_percep is not None:
                l_total += (l_percep * 0.2)
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += (l_style * 0.2)
                loss_dict['l_style'] = l_style
            if l_contrast_percep is not None:
                l_total += l_contrast_percep
                loss_dict['l_contrast_percep'] = l_contrast_percep
            if l_contrast_style is not None:
                l_total += l_contrast_style
                loss_dict['l_contrast_style'] = l_contrast_style

        ###########################################

        l_total.backward()
        self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)





