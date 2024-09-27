import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import LOSS_REGISTRY
import torchvision.transforms as transforms
import numpy as np
from basicsr.archs.vgg_arch import VGGFeatureExtractor

import pyiqa

# @LOSS_REGISTRY.register()
# class CharbonnierLoss(nn.Module):
#     """Charbonnier Loss (L1)"""
#
#     def __init__(self, eps=1e-3):
#         super(CharbonnierLoss, self).__init__()
#         self.eps = eps
#
#     def forward(self, x, y):
#         diff = x - y
#         # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
#         loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
#         return loss

# @LOSS_REGISTRY.register()
# class EdgeLoss(nn.Module):
#     def __init__(self):
#         super(EdgeLoss, self).__init__()
#         k = torch.Tensor([[.05, .25, .4, .25, .05]])
#         self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
#         if torch.cuda.is_available():
#             self.kernel = self.kernel.cuda()
#         self.loss = CharbonnierLoss()
#
#     def conv_gauss(self, img):
#         n_channels, _, kw, kh = self.kernel.shape
#         img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
#         return F.conv2d(img, self.kernel, groups=n_channels)
#
#     def laplacian_kernel(self, current):
#         filtered    = self.conv_gauss(current)    # filter
#         down        = filtered[:,:,::2,::2]               # downsample
#         new_filter  = torch.zeros_like(filtered)
#         new_filter[:,:,::2,::2] = down*4                  # upsample
#         filtered    = self.conv_gauss(new_filter) # filter
#         diff = current - filtered
#         return diff
#
#     def forward(self, x, y):
#         loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
#         return loss


@LOSS_REGISTRY.register()
class LPIPSLoss(nn.Module):
    """LPIPS loss with vgg backbone.
    """
    def __init__(self, loss_weight = 1.0):
        super(LPIPSLoss, self).__init__()
        self.model = pyiqa.create_metric('lpips-vgg', as_loss=True)
        self.loss_weight = loss_weight

    def forward(self, x, gt):
        return self.model(x, gt) * self.loss_weight, None


@LOSS_REGISTRY.register()
class KDLoss(nn.Module):
    """
    Args:
        loss_weight (float): Loss weight for KD loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, temperature = 0.15):
        super(KDLoss, self).__init__()
    
        self.loss_weight = loss_weight
        self.temperature = temperature

    def forward(self, S1_fea, S2_fea):
        """
        Args:
            S1_fea (List): contain shape (N, L) vector. 
            S2_fea (List): contain shape (N, L) vector.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        loss_KD_dis = 0
        loss_KD_abs = 0
        for i in range(len(S1_fea)):
            S2_distance = F.log_softmax(S2_fea[i] / self.temperature, dim=1)
            S1_distance = F.softmax(S1_fea[i].detach()/ self.temperature, dim=1)
            loss_KD_dis += F.kl_div(
                        S2_distance, S1_distance, reduction='batchmean')
            loss_KD_abs += nn.L1Loss()(S2_fea[i], S1_fea[i].detach())
        return self.loss_weight * loss_KD_dis, self.loss_weight * loss_KD_abs


@LOSS_REGISTRY.register()
class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


@LOSS_REGISTRY.register()
class BCPLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(BCPLoss, self).__init__()
        self.loss_weight = loss_weight

    def get_bright_channel(self, I, w):
        _, _, H, W = I.shape
        maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
        bc = maxpool(I[:, :, :, :])

        return bc

    def get_atmosphere(self, I, bright_ch, p):
        B, _, H, W = bright_ch.shape
        num_pixel = int(p * H * W)
        flat_bc = bright_ch.resize(B, H * W)
        flat_I = I.resize(B, 3, H * W)
        index = torch.argsort(flat_bc, descending=False)[:, :num_pixel]
        A = torch.zeros((B, 3)).to('cuda')

        for i in range(B):
            A[i] = flat_I[i, :, index].mean((1, 2))

        return A

    def forward(self, img, T, w=35, p=0.0001):
        bright_channel = self.get_bright_channel(img, w)

        A = self.get_atmosphere(img, bright_channel, p)

        norm_I = (1 - img) / (1 - A[:, :, None, None] + 1e-6)
        bright_channel = self.get_bright_channel(norm_I, w)
        t_slide = (1 - 0.95 * bright_channel)

        loss = F.smooth_l1_loss(T, t_slide) * self.loss_weight

        return loss


@LOSS_REGISTRY.register()
class DCPLoss(nn.Module):
    def __init__(self, loss_weight=1.0, w=15, p=0.001, omega=0.95, eps=1e-5, lambda1=2, lambda2=1e-2):
        super(DCPLoss, self).__init__()
        self.loss_weight = loss_weight
        self.w = w
        self.p = p
        self.omega = omega
        self.eps = eps
        self.param = [lambda1, lambda2]

    def get_dark_channel(self, I, w):
        _, _, H, W = I.shape
        maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
        dc = maxpool(0 - I[:, :, :, :])

        return -dc

    def get_atmosphere(self, I, dark_ch, p):
        B, _, H, W = dark_ch.shape
        num_pixel = int(p * H * W)
        flat_dc = dark_ch.resize(B, H * W)
        flat_I = I.resize(B, 3, H * W)
        index = torch.argsort(flat_dc, descending=True)[:, :num_pixel]
        A = torch.zeros((B, 3)).to('cuda')

        for i in range(B):
            A[i] = flat_I[i, :, index[i][torch.argsort(torch.max(flat_I[i][:, index[i]], 0)[0], descending=True)[0]]]

        return A

    def forward(self, img, y_pred):
        dark_channel = self.get_dark_channel(img, self.w)
        A = self.get_atmosphere(img, dark_channel, self.p)

        normI = img.permute(2, 3, 0, 1)
        normI = (normI / A).permute(2, 3, 0, 1)
        norm_dc = self.get_dark_channel(normI, self.w)

        t_slide = (1 - self.omega * norm_dc)  # transmission map predicted by dark channel prior

        patches_I = F.unfold(y_pred, (3, 3))
        Y_I = patches_I.repeat([1, 9, 1])
        Y_J = patches_I.repeat_interleave(9, 1)

        temp = F.unfold(img, (3, 3))
        B, N = temp.shape[0], temp.shape[2]

        img_patches = temp.view(B, 3, 9, N).permute(0, 3, 2, 1)
        mean_patches = torch.mean(img_patches, 2, True)

        XX_T = (torch.matmul(img_patches.permute(0, 1, 3, 2), img_patches) / 9)
        UU_T = torch.matmul(mean_patches.permute(0, 1, 3, 2), mean_patches)
        var_patches = XX_T - UU_T

        matrix_to_invert = (self.eps / 9) * torch.eye(3).to('cuda') + var_patches
        var_fac = torch.inverse(matrix_to_invert)

        weights = torch.matmul(img_patches - mean_patches, var_fac)
        weights = torch.matmul(weights, (img_patches - mean_patches).permute(0, 1, 3, 2)) + 1
        weights = weights / 9
        weights = weights.view(-1, N, 81)

        neighbour_difference = (Y_I - Y_J) ** 2
        fidelity_term = torch.matmul(neighbour_difference, weights).sum()  # data fidelity
        prior_term = ((y_pred - t_slide) ** 2).sum()  # regularization

        loss = (self.param[0] * (1 / 2) * fidelity_term + self.param[1] * prior_term) / N
        loss = loss * self.loss_weight

        return loss


@LOSS_REGISTRY.register()
class ColorLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(ColorLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, img):
        b, c, h, w = img.shape

        mean_rgb = torch.mean(img, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.mean(torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5))

        loss = k * self.loss_weight

        return loss


@LOSS_REGISTRY.register()
class SpatialLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(SpatialLoss, self).__init__()
        self.loss_weight = loss_weight
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = torch.mean((D_left + D_right + D_up + D_down))

        loss = E * self.loss_weight

        return loss

@LOSS_REGISTRY.register()
class ExposureLoss(nn.Module):
    def __init__(self, patch_size, mean_val, loss_weight=1.0):
        super(ExposureLoss, self).__init__()
        self.loss_weight = loss_weight
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        loss = d * self.loss_weight
        return loss


@LOSS_REGISTRY.register()
class TVLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(TVLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

@LOSS_REGISTRY.register()
class KLLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(KLLoss, self).__init__()
        self.loss_weight = loss_weight
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, p, q, pad_mask=None):
        p = F.log_softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        loss = self.kl_loss(p,q) * self.loss_weight
        return loss


@LOSS_REGISTRY.register()
class ContrastPerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(ContrastPerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def standard_perceptual_loss(self,x,gt):
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def forward(self, x, gt, ng):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())
        ng_features = self.vgg(ng.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += (self.criterion(x_features[k], gt_features[k])/self.criterion(x_features[k],ng_features[k])) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += (self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k]))/self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        ng_features[k]))) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
