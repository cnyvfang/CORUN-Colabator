import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from CORUN_Colabator.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel
from CORUN_Colabator.models import lr_scheduler as lr_scheduler

from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.nn as nn

from basicsr.utils.dist_util import master_only
import os
import time
import cv2
import pyiqa

from torch.nn import functional as F


class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        # define network

        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g, find_unused_parameters=opt.get('find_unused_parameters', False))
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.net_g.train()
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('psnr_opt'):
            self.cri_psnr = build_loss(train_opt['psnr_opt']).to(self.device)
        else:
            self.cri_psnr = None

        if train_opt.get('bcp_opt'):
            self.cri_bcp = build_loss(train_opt['bcp_opt']).to(self.device)
        else:
            self.cri_bcp = None

        if train_opt.get('dcp_opt'):
            self.cri_dcp = build_loss(train_opt['dcp_opt']).to(self.device)
        else:
            self.cri_dcp = None

        if train_opt.get('color_opt'):
            self.cri_color = build_loss(train_opt['color_opt']).to(self.device)
        else:
            self.cri_color = None

        if train_opt.get('spatial_opt'):
            self.cri_spatial = build_loss(train_opt['spatial_opt']).to(self.device)
        else:
            self.cri_spatial = None

        if train_opt.get('exp_opt'):
            self.cri_exp = build_loss(train_opt['exp_opt']).to(self.device)
        else:
            self.cri_exp = None

        if train_opt.get('tv_opt'):
            self.cri_tv = build_loss(train_opt['tv_opt']).to(self.device)
        else:
            self.cri_tv = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        if train_opt.get('mse_opt'):
            self.cri_mse = build_loss(train_opt['mse_opt']).to(self.device)

        if train_opt.get('kl_opt'):
            self.cri_kl = build_loss(train_opt['kl_opt']).to(self.device)

        if train_opt.get('contrastperceptual_opt'):
            self.cri_contrastperceptual = build_loss(train_opt['contrastperceptual_opt']).to(self.device)

        if train_opt.get('edge_opt'):
            self.cri_edge = build_loss(train_opt['edge_opt']).to(self.device)

        if train_opt.get('charbonnier_opt'):
            self.cri_charbonnier = build_loss(train_opt['charbonnier_opt']).to(self.device)

        self.setup_optimizers()

    def model_to_device(self, net, find_unused_parameters=False):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        net = net.to(self.device)
        if self.opt['dist']:
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    @master_only
    def print_each_log_vars(self):
        for k, v in self.log_vars.items():
            print(k,": ", (torch.exp(-v)).item(), end='\t')
        print("\n")

    def setup_optimizers(self):
        train_opt = self.opt['train']
        gen_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                gen_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, gen_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        scheduler_opt = train_opt['gen_scheduler']
        scheduler_type = scheduler_opt.pop('type')
        self.setup_schedulers(self.optimizer_g, scheduler_opt, scheduler_type)


    def setup_schedulers(self, optimizer, scheduler_opt, scheduler_type):
        """Set up schedulers."""
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            self.schedulers.append(
                    lr_scheduler.MultiStepRestartLR(optimizer,
                                                    **scheduler_opt))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartLR(
                        optimizer, **scheduler_opt))
        elif scheduler_type == 'CosineAnnealingWarmupRestarts':
            self.schedulers.append(
                    lr_scheduler.CosineAnnealingWarmupRestarts(
                        optimizer, **scheduler_opt))
        elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
            self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartCyclicLR(
                        optimizer, **scheduler_opt))
        elif scheduler_type == 'TrueCosineAnnealingLR':
            print('..', 'cosineannealingLR')
            self.schedulers.append(
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_opt))
        elif scheduler_type == 'CosineAnnealingLRWithRestart':
            print('..', 'CosineAnnealingLR_With_Restart')
            self.schedulers.append(
                    lr_scheduler.CosineAnnealingLRWithRestart(optimizer, **scheduler_opt))
        elif scheduler_type == 'LinearLR':
            self.schedulers.append(
                    lr_scheduler.LinearLR(
                        optimizer, **scheduler_opt))
        elif scheduler_type == 'VibrateLR':
            self.schedulers.append(
                    lr_scheduler.VibrateLR(
                        optimizer, **scheduler_opt))
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


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
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)


    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        with torch.no_grad():
            dataset_name = dataloader.dataset.opt['name']
            with_metrics = self.opt['val'].get('metrics') is not None
            use_pbar = self.opt['val'].get('pbar', False)
            save_source_img = self.opt['val'].get('save_source', False)

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

                visuals = self.get_current_visuals(current_iter)

                sr_img = tensor2img([visuals['result']])
                lq_img = tensor2img([visuals['lq']])
                gt_img = tensor2img([visuals['gt']])

                # tentative for out of GPU memory

                del self.output_transmissions
                del self.step_images
                del self.recon_images
                del self.lq
                del self.gt
                del self.output
                del self.outputs

                torch.cuda.empty_cache()

                if save_img:
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                 f'{img_name}_{current_iter}.png')
                        save_lq_path = osp.join(self.opt['path']['visualization'], img_name,
                                                f'{img_name}_lq.png')

                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')
                        if save_source_img:
                            save_lq_path = osp.join(self.opt['path']['visualization'], dataset_name, 'lq',
                                                    f'{img_name}_lq.png')
                            save_gt_path = osp.join(self.opt['path']['visualization'], dataset_name, 'gt',
                                                    f'{img_name}_lq.png')

                    imwrite(sr_img, save_img_path)
                    if save_source_img:
                        imwrite(lq_img, save_lq_path)
                        imwrite(gt_img, save_gt_path)

                    metric_data['nima'] = nima
                    metric_data['brisque'] = brisque
                    metric_data['img'] = save_img_path
                    metric_data['img2'] = save_gt_path
                    metric_data['lq'] = save_lq_path

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
                    # update the best metric result
                    self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def get_current_visuals(self, current_iter):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        # if hasattr(self, 'gt'):
        out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    @master_only
    def save_training_state(self, epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        if current_iter != -1:
            state = {'epoch': epoch, 'iter': current_iter, 'optimizers': [], 'schedulers': []}
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
            save_filename = f'{current_iter}.state'
            save_path = os.path.join(self.opt['path']['training_states'], save_filename)

            # avoid occasional writing errors
            retry = 3
            while retry > 0:
                try:
                    torch.save(state, save_path)
                except Exception as e:
                    logger = get_root_logger()
                    logger.warning(f'Save training state error: {e}, remaining retry times: {retry - 1}')
                    time.sleep(1)
                else:
                    break
                finally:
                    retry -= 1
            if retry == 0:
                logger.warning(f'Still cannot save {save_path}. Just ignore it.')
                # raise IOError(f'Cannot save {save_path}.')

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    @master_only
    def save_memory_bank(self, current_iter):
        self.memory_bank.save(self.opt['path']['models'], current_iter)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)

        self.save_training_state(epoch, current_iter)

    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.opt['dist']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt['rank'] == 0:
                    losses /= self.opt['world_size']
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict
