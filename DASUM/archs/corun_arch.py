
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import to_2tuple, trunc_normal_
import torch.nn.functional as F
from torchvision import transforms

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


def shift_back(inputs,step=2):          # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]

class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b,h*w,c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b,h,w,c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class MST(nn.Module):
    def __init__(self, in_dim=30, out_dim=30, dim=30, stage=2, num_blocks=[2,4,4]):
        super(MST, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                # nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                Downsample(dim_stage),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = MSAB(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                # nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                Upsample(dim_stage),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, y=None):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x) if y is None else self.embedding(torch.cat([x, y], dim=1))

        # Encoder
        fea_encoder = []
        for (MSAB, FeaDownSample) in self.encoder_layers:
            fea = MSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1))
            fea = LeWinBlcok(fea)

        # Mapping
        out = self.mapping(fea) + x

        return out


class ProximalTransmissionBlock(nn.Module):
    def __init__(self):
        super(ProximalTransmissionBlock, self).__init__()
        self.step_lambda = nn.Parameter(F.sigmoid(torch.Tensor([0.3]))*1.0)


    def forward(self, p, q_prev, t_prev):
        # p: hazy image
        # q_prev: previous dehazed image (q_hat)
        # t_prev: previous transmission map (t)
        one = torch.ones_like(p)
        epsilon = 0.0000001 # to avoid potential division by zero

        storedValue = (one - q_prev)
        storedValue = torch.masked_fill(storedValue, storedValue == 0, epsilon)

        factor_1 = (one - p + (self.step_lambda * t_prev / storedValue))
        factor_1 = torch.sum(factor_1, dim=1, keepdim=True)

        factor_2 = (one - q_prev + (self.step_lambda / storedValue))
        factor_2 = torch.sum(factor_2, dim=1, keepdim=True)
        factor_2 = torch.masked_fill(factor_2, factor_2 == 0, epsilon)

        t_hat = factor_1 / factor_2

        t_hat = t_hat.clamp(0.05, 0.999)

        return t_hat


class ProximalDehazingBlock(nn.Module):
    def __init__(self):
        super(ProximalDehazingBlock, self).__init__()
        self.step_lambda = nn.Parameter(F.sigmoid(torch.Tensor([0.3]))*1.0)

    def forward(self, p, q_prev, t):
        # p: hazy image
        # q_t_prev: previous dehazed image
        # t_t_prev: previous transmission map

        ## GDM
        one = torch.ones_like(p)

        epsilon = 0.0000001 # to avoid potential division by zero

        storedValue = t * t
        factor_1 = (t * p + storedValue - t + self.step_lambda * q_prev)
        factor_2 = (storedValue + self.step_lambda * one)
        factor_2 = torch.masked_fill(factor_2, factor_2 == 0, epsilon)

        q_hat = factor_1 / factor_2

        q_hat = q_hat.clamp(0.0, 1.0)

        return q_hat


class Basic_block_fix_Plus(nn.Module):
    def __init__(self, last=False):
        super(Basic_block_fix_Plus, self).__init__()

        self.trans_prox = ProximalTransmissionBlock()
        if not last:
            self.trans_net = MST(in_dim=4, out_dim=1, dim=30, stage=3, num_blocks=[1,1,1])

        self.dehaze_prox = ProximalDehazingBlock()
        self.dehaze_net = MST(in_dim=4, out_dim=3, dim=30, stage=3, num_blocks=[1,1,1])

    def forward(self, img, stage1_trans, stage1_img, stage1_trans_hat, stage1_img_hat, last=False):
        ## GDM
        stage2_trans_hat = self.trans_prox(img, stage1_img_hat, stage1_trans)
        if not last:
            stage2_trans = self.trans_net(stage2_trans_hat, stage1_img)
        else:
            stage2_trans = stage2_trans_hat

        stage2_img_hat = self.dehaze_prox(img, stage1_img, stage2_trans_hat)
        stage2_img = self.dehaze_net(stage2_img_hat, stage2_trans_hat)

        return stage2_trans, stage2_img, stage2_trans_hat, stage2_img_hat

@ARCH_REGISTRY.register()
class CORUN(nn.Module):
    def __init__(self, depth=3):
        super(CORUN, self).__init__()

        self.depth = depth - 2
        self.basic = nn.Sequential(*[Basic_block_fix_Plus() for _ in range(self.depth)])
        self.first = Basic_block_fix_Plus()
        self.last = Basic_block_fix_Plus(last=True)

    def forward(self, img, debug=False, t_mode=False, finetune=False):

        res = []
        disp_tres = []
        ires = []
        recon_res = []
        t_hat = []

        cacl_atmosphere = 1.0
        # cacl_transmission = self.atmosphere.get_transmission_map_with_a(img, cacl_atmosphere)
        img = img / cacl_atmosphere
        init_q = img
        init_t = torch.ones(img.shape[0], 1, img.shape[2], img.shape[3]).to(img.device) - 0.5

        stage1_trans = init_t
        stage1_trans_hat = init_t
        stage1_img = init_q
        stage1_img_hat = init_q

        tres = torch.zeros_like(init_t)

        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------

        stage1_trans, stage1_img, stage1_trans_hat, stage1_img_hat = self.first(img, stage1_trans, stage1_img,
                                                                                stage1_trans_hat, stage1_img_hat)
        res.append(stage1_img)
        if debug:
            disp_tres.append(torch.cat([stage1_trans_hat, stage1_trans], 3))
            ires.append(torch.cat([stage1_img_hat, stage1_img], 3))
            recon_res.append((stage1_img * stage1_trans_hat + (1 - stage1_trans_hat)).clamp(0.0, 1.0))
        if t_mode:
            tres = tres + (1-stage1_trans_hat)
        if finetune:
            t_hat.append(stage1_trans_hat)


        ##-------------------------------------------
        ##-------------- Stage 2-6 ------------------
        ##-------------------------------------------

        for i in range(self.depth):
            stage1_trans, stage1_img, stage1_trans_hat, stage1_img_hat = self.basic[i](img, stage1_trans, stage1_img,
                                                                                    stage1_trans_hat, stage1_img_hat)
            res.append(stage1_img)
            if debug:
                disp_tres.append(torch.cat([stage1_trans_hat, stage1_trans], 3))
                ires.append(torch.cat([stage1_img_hat, stage1_img], 3))
                recon_res.append((stage1_img * stage1_trans_hat + (1 - stage1_trans_hat)).clamp(0.0, 1.0))
            if t_mode:
                tres = tres + (1-stage1_trans_hat)
            if finetune:
                t_hat.append(stage1_trans_hat)

        ##-------------------------------------------
        ##-------------- Stage Last -----------------
        ##-------------------------------------------

        stage1_trans, stage1_img, stage1_trans_hat, stage1_img_hat = self.last(img, stage1_trans, stage1_img,
                                                                               stage1_trans_hat, stage1_img_hat, last=True)
        res.append(stage1_img)
        if debug:
            disp_tres.append(torch.cat([stage1_trans_hat, stage1_trans], 3))
            ires.append(torch.cat([stage1_img_hat, stage1_img], 3))
            recon_res.append((stage1_img * stage1_trans_hat + (1 - stage1_trans_hat)).clamp(0.0, 1.0))
        if t_mode:
            tres = tres + (1-stage1_trans_hat)
        if finetune:
            t_hat.append(stage1_trans_hat)

        if debug:
            return res[::-1], disp_tres[::-1], ires[::-1], recon_res[::-1]

        if t_mode:
            recon_I = (stage1_img * (1-tres) + (1 - (1-tres))).clamp(0, 1)
            recon_I_Gray = transforms.Grayscale(num_output_channels=1)(recon_I)
            return res[::-1], recon_I_Gray

        if finetune:
            return res[::-1], t_hat[::-1]

        return res[::-1]