import model.backbone.resnet as resnet
from model.backbone.xception import xception
from segmentation_models_pytorch import Unet
import numpy as np
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F


def _cfg_bool(cfg, key, default=False):
    value = cfg.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y")


class WaveletRoadDetailEnhancement(nn.Module):
    """Wavelet-guided road detail enhancement.

    The block decomposes decoder features into Haar wavelet sub-bands, enhances
    directional high-frequency components (LH/HL/HH), reconstructs a spatial
    detail feature with IDWT, and injects it back into the semantic feature.
    This is designed for thin roads, road boundaries and discontinuous road
    segments in remote sensing segmentation.
    """

    def __init__(self, channels):
        super().__init__()
        self.hf_enhance = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False),
            nn.BatchNorm2d(channels * 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 3, channels * 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels * 3),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            # Gate informed by both original feature (x) and high-freq detail
            # so it can suppress noise while preserving genuine road structure.
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    @staticmethod
    def haar_dwt(x):
        _, _, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (-x00 - x01 + x10 + x11) * 0.5
        hl = (-x00 + x01 - x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh, pad_h, pad_w

    @staticmethod
    def haar_idwt(ll, lh, hl, hh, pad_h=0, pad_w=0):
        x00 = (ll - lh - hl + hh) * 0.5
        x01 = (ll - lh + hl - hh) * 0.5
        x10 = (ll + lh - hl - hh) * 0.5
        x11 = (ll + lh + hl + hh) * 0.5

        b, c, h, w = ll.shape
        x = torch.zeros(b, c, h * 2, w * 2, device=ll.device, dtype=ll.dtype)
        x[:, :, 0::2, 0::2] = x00
        x[:, :, 0::2, 1::2] = x01
        x[:, :, 1::2, 0::2] = x10
        x[:, :, 1::2, 1::2] = x11

        if pad_h:
            x = x[:, :, :-pad_h, :]
        if pad_w:
            x = x[:, :, :, :-pad_w]
        return x

    def forward(self, x):
        ll, lh, hl, hh, pad_h, pad_w = self.haar_dwt(x)
        hf = torch.cat([lh, hl, hh], dim=1)
        hf = self.hf_enhance(hf)
        lh, hl, hh = torch.chunk(hf, chunks=3, dim=1)
        detail = self.haar_idwt(ll, lh, hl, hh, pad_h, pad_w)

        # Gate jointly informed by original feature x and reconstructed detail
        gate = self.gate(torch.cat([x, detail], dim=1))
        # Inject only the gated high-frequency delta to avoid redundant x copy
        return self.fuse(torch.cat([x, detail * gate], dim=1))


class SemanticFrequencyFusionAttention(nn.Module):
    """Semantic-frequency fusion attention adapted from WFA-SRNet FA.

    Semantic road features suppress irrelevant high-frequency background edges,
    while high-frequency cues emphasize boundaries and thin road structures.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)

        # Avg + Max pooling branches for richer channel attention (CBAM-style)
        self.sem_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        self.sem_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        self.freq_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        self.freq_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, semantic_feature, frequency_feature):
        # Avg + Max for richer channel attention
        semantic_weight = torch.sigmoid(
            self.sem_avg(semantic_feature) + self.sem_max(semantic_feature)
        )
        frequency_weight = torch.sigmoid(
            self.freq_avg(frequency_feature) + self.freq_max(frequency_feature)
        )

        semantic_guided = semantic_feature * frequency_weight
        frequency_guided = frequency_feature * semantic_weight
        return self.fuse(torch.cat([semantic_guided, frequency_guided], dim=1))


class PatchNet(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        if cfg['model'] == 'deeplabv3plus':
            self.segnet = DeepLabV3Plus(cfg)
        if cfg['model'] == 'unet':
            encoder_weights = cfg.get('encoder_weights', 'imagenet' if self.training else None)
            self.segnet = Unet(encoder_name=cfg['backbone'], classes=cfg.get('nclass', 2), encoder_weights=encoder_weights)

        self.aug = str(cfg.get('training_config', {}).get('use_aug', True)).lower() == 'true'
        self.use_wavelet = _cfg_bool(cfg, 'use_wavelet', True)
        # Fine-grained ablation switches (only effective when use_wavelet=True)
        self.use_wrde = self.use_wavelet and _cfg_bool(cfg, 'use_wrde', True)
        self.use_sffa = self.use_wavelet and _cfg_bool(cfg, 'use_sffa', True)

        decoder_channels = self._infer_decoder_channels()
        if self.use_wavelet:
            if self.use_wrde:
                self.wavelet_enhance = WaveletRoadDetailEnhancement(decoder_channels)
            if self.use_sffa:
                self.semantic_frequency_fusion = SemanticFrequencyFusionAttention(
                    decoder_channels,
                    reduction=cfg.get('freq_attention_reduction', 16),
                )
            self.freq_head = nn.Conv2d(decoder_channels, cfg.get('nclass', 2), kernel_size=1, bias=True)

    def _infer_decoder_channels(self):
        if self.cfg['model'] == 'deeplabv3plus':
            return 256
        if self.cfg['model'] == 'unet':
            # segmentation_models_pytorch SegmentationHead is a Sequential-like
            # module whose first layer is the final Conv2d.
            return self.segnet.segmentation_head[0].in_channels
        raise NotImplementedError('Unsupported model: {}'.format(self.cfg['model']))

    def _decode_feature(self, img):
        if self.cfg['model'] == 'deeplabv3plus':
            return self.segnet(img)
        if self.cfg['model'] == 'unet':
            fea = self.segnet.encoder(img)
            return self.segnet.decoder(fea)
        raise NotImplementedError('Unsupported model: {}'.format(self.cfg['model']))

    def _classify_feature(self, fea, out_size=None):
        if self.cfg['model'] == 'deeplabv3plus':
            out = self.segnet.classifier(fea)
            if out_size is not None:
                out = F.interpolate(out, size=out_size, mode="bilinear", align_corners=True)
            return out
        if self.cfg['model'] == 'unet':
            return self.segnet.segmentation_head(fea)
        raise NotImplementedError('Unsupported model: {}'.format(self.cfg['model']))

    def _apply_wavelet_fusion(self, fea):
        if not self.use_wavelet:
            return fea, fea, None

        # WRDE: wavelet-guided road detail enhancement
        fea_freq = self.wavelet_enhance(fea) if self.use_wrde else fea
        # SFFA: semantic-frequency fusion attention
        fea_fused = self.semantic_frequency_fusion(fea, fea_freq) if self.use_sffa else fea_freq
        out_freq = self.freq_head(fea_freq)
        return fea_fused, fea_freq, out_freq

    def forward(self, x):
        if self.training:
            img_ori, img_u, img_u_s1, img_u_s2 = x
            h, w = img_ori.shape[-2:]
            img = torch.cat((img_ori, img_u))
            img_h = torch.cat((img_ori, img_u_s1))
            img_v = torch.cat((img_ori, img_u_s2))
        else:
            img = x
            h, w = img.shape[-2:]

        fea = self._decode_feature(img)
        fea_fused, fea_freq, out_freq = self._apply_wavelet_fusion(fea)
        out = self._classify_feature(fea_fused, out_size=(h, w))

        if self.training:
            if out_freq is None:
                out_freq = out
            elif self.cfg['model'] == 'deeplabv3plus':
                out_freq = F.interpolate(out_freq, size=(h, w), mode="bilinear", align_corners=True)

            fea_h, fea_v, out_h, out_v = self._strip_forward(img_h, img_v)
            return out, fea_h, fea_v, fea_freq, out_h, out_v, out_freq
        else:
            return out

    def _strip_forward(self, img_h, img_v):

        fea_h, out_h = self.forward_puzzles(img_h, p=0)
        fea_v, out_v = self.forward_puzzles(img_v, p=1)
        return fea_h, fea_v, out_h, out_v

    def puzzle(self, x, p, grid):
        if p == 0:
            fea = rearrange(x, 'b c (n h) (m w) -> (n m b) c h w', n=grid*2, m=2)
        if p == 1:
            fea = rearrange(x, 'b c (n h) (m w) -> (n m b) c h w', n=2, m=grid*2)
        return fea, p

    def recover_puzzle(self,x, grid, p):
        if p == 0:
            fea = rearrange(x, '(n m b) c h w -> b c (n h) (m w)', n=grid*2, m=2)
        if p == 1:
            fea = rearrange(x, '(n m b) c h w -> b c (n h) (m w)', n=2, m=grid*2)
        return fea

    def forward_puzzles(self, x, p):
        h, w = x.shape[-2:]
        if self.aug:
            data = [0, 32, 64]
            pad = 2*np.random.choice(data)
            if p == 0:
                padding = [0, 0, pad, pad]
            elif p == 1:
                padding = [pad, pad, 0, 0]
            x = F.pad(x, padding, 'constant', 0)

        else:
            pad = 0

        x, puzzle_types = self.puzzle(x, p, 2)
        fea = self._decode_feature(x)

        re_fea = self.recover_puzzle(fea, 2, puzzle_types)
        if self.cfg['model'] == 'deeplabv3plus':
            pad = int(pad / 4)
        if pad > 0:
            if p == 0:
                re_fea = re_fea[:, :, pad:-pad,:]
            elif p == 1:
                re_fea = re_fea[:, :, :, pad:-pad]

        re_fea, _, _ = self._apply_wavelet_fusion(re_fea)
        out = self._classify_feature(re_fea, out_size=(h, w))

        return re_fea, out

class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3Plus, self).__init__()

        if 'resnet' in cfg['backbone']:
            self.backbone = resnet.__dict__[cfg['backbone']](pretrained=True,
                                                             replace_stride_with_dilation=cfg['replace_stride_with_dilation'])
        else:
            assert cfg['backbone'] == 'xception'
            self.backbone = xception(pretrained=True)

        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, cfg['dilations'])

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)

    def forward(self, x, need_fp=False):

        h, w = x.shape[-2:]

        feats = self.backbone.base_forward(x)
        c1, c4 = feats[0], feats[-1]

        if need_fp:
            outs = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))),
                                torch.cat((c4, nn.Dropout2d(0.5)(c4))))
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)

            return out, out_fp

        feature = self._decode(c1, c4)
        return feature

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)

        return feature


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)
