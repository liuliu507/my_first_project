# fadc.py
"""
Frequency-Adaptive Dilated Convolution (FADC)
Implements AdaDR (adaptive dilation via soft selection of dilations),
AdaKern (low/high frequency kernel branches with per-channel dynamic weights),
and FreqSelect (frequency-band decomposition + spatial selection maps).

Notes on implementation choices:
- AdaDR in the paper predicts a per-pixel dilation D̂(p) (integer). Standard torch.nn.Conv2d
  cannot accept per-pixel integer dilations. To keep the module differentiable and practical,
  we implement AdaDR as several parallel convolution branches with different dilation rates
  (dilation_list) and a predicted per-pixel soft weight for each branch (softmax across branches).
  This is a standard, faithful and differentiable approximation which preserves the trade-off
  between receptive field and bandwidth described in the paper.
- AdaKern: implemented as two parallel paths:
    * low-frequency path: local avg (or small low-pass) followed by conv (1x1 or kxk)
    * high-frequency path: input - lowpass then conv
  Per-channel gating λ_l and λ_h are predicted by global pooling -> MLP.
- FreqSelect: performs FFT-based band masking into B bands (octave-wise by default),
  computes per-band spatial selection maps via small convs, and recomposes the feature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----------------------------
# Frequency Selection Module
# ----------------------------
class FreqSelect(nn.Module):
    """
    Decompose input into frequency bands (via FFT masks), predict spatial selection
    maps for each band, reweight and reconstruct spatial feature.
    B default = 4 bands: [0,1/16), [1/16,1/8), [1/8,1/4), [1/4,1/2]
    """

    def __init__(self, in_ch, k_list=None, act='sigmoid', spatial_kernel=3, init_zero=True):
        super().__init__()
        # octave-wise default thresholds (normalized freq; we use mask via center cropping)
        # The actual masks are created based on input size in forward
        self.in_ch = in_ch
        self.B = 4
        self.act = act
        # per-band spatial selectors: output 1 channel per-group. We use group = 1 (can be adapted)
        self.spatial_convs = nn.ModuleList([
            nn.Conv2d(in_ch, in_ch, kernel_size=spatial_kernel, padding=spatial_kernel // 2, groups=1, bias=True)
            for _ in range(self.B)
        ])
        # optional lowfreq attention channel (paper uses lowfreq too)
        self.lowfreq_conv = nn.Conv2d(in_ch, in_ch, kernel_size=spatial_kernel, padding=spatial_kernel//2, groups=1, bias=True)

        if init_zero:
            for conv in self.spatial_convs:
                nn.init.constant_(conv.weight, 0.0)
                nn.init.constant_(conv.bias, 0.0)
            nn.init.constant_(self.lowfreq_conv.weight, 0.0)
            nn.init.constant_(self.lowfreq_conv.bias, 0.0)

    def sp_act(self, x):
        if self.act == 'sigmoid':
            return torch.sigmoid(x) * 2.0  # paper multiplies/uses scaled sigmoid in places
        elif self.act == 'softmax':
            # softmax across band dimension expected outside; here treat per-band conv separately
            return x
        else:
            return x

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: same shape, frequency-balanced feature
        """
        b, c, H, W = x.shape
        # FFT (complex) with shift
        x_fft = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'), dim=(-2, -1))  # (b,c,H,W) complex
        # compute masks for 4 bands: square masks centered at (H/2,W/2)
        # bands defined by radii fractions relative to Nyquist (0.5 normalized); but we use indices
        # We'll produce masks by cropping square windows of sizes proportional to bands
        # band1: very low freq (center small window)
        # band2..band4: increasing windows
        # define radii in pixels
        # normalized radii as fractions of half-size
        r_half_h = H // 2
        r_half_w = W // 2
        # Fractions chosen to match octave-like division: [0,1/16), [1/16,1/8), [1/8,1/4), [1/4,1/2]
        frac = [1/16.0, 1/8.0, 1/4.0, 1/2.0]
        radii = [max(1, int(fr * min(H, W) / 2.0)) for fr in frac]  # pixel radii
        # Create cumulative masks and extract band masks
        center_h = H // 2
        center_w = W // 2
        masks = []
        prev = 0
        for r in radii:
            mask = torch.zeros((H, W), dtype=x_fft.real.dtype, device=x.device)
            h0 = max(0, center_h - r)
            h1 = min(H, center_h + r)
            w0 = max(0, center_w - r)
            w1 = min(W, center_w + r)
            mask[h0:h1, w0:w1] = 1.0
            masks.append(mask)
        # band masks: band0 = masks[0], band1 = masks[1] - masks[0], ...
        band_masks = []
        last = torch.zeros_like(masks[0])
        for m in masks:
            band_masks.append((m - last).clamp(0.0, 1.0))
            last = m
        # highest band: remainder up to full
        rem = torch.ones_like(masks[0]) - last
        band_masks.append(rem)  # now length 5, but we want exactly 4 bands; we will combine last two
        # Combine last two so we have 4 bands:
        if len(band_masks) == 5:
            band_masks[3] = band_masks[3] + band_masks[4]
            band_masks = band_masks[:4]

        # prepare complex x_fft as two real tensors
        x_fft_real = x_fft.real
        x_fft_imag = x_fft.imag

        x_list = []
        pre_x = x
        for idx, mask in enumerate(band_masks):
            # apply mask (broadcast over batch & channels)
            mask_tensor = mask.unsqueeze(0).unsqueeze(0)  # 1,1,H,W
            # masked freq domain
            xf_r_masked = x_fft_real * mask_tensor
            xf_i_masked = x_fft_imag * mask_tensor
            xf_masked = torch.complex(xf_r_masked, xf_i_masked)
            # inverse shift+ifft to get spatial low part
            xf_ifftshift = torch.fft.ifftshift(xf_masked, dim=(-2, -1))
            low_part = torch.fft.ifft2(xf_ifftshift, norm='ortho').real  # keep real part
            high_part = pre_x - low_part
            pre_x = low_part
            # spatial selector for this band
            sel = self.spatial_convs[idx](x)  # (b,c,H,W)
            sel = self.sp_act(sel)
            # apply selector: per-channel spatial gating (we multiply channelwise)
            tmp = sel * high_part
            x_list.append(tmp)

        # lowfreq attention (last low part)
        low_sel = self.lowfreq_conv(x)
        low_sel = self.sp_act(low_sel)
        x_list.append(low_sel * pre_x)

        out = sum(x_list)
        return out

# ----------------------------
# AdaKern (Adaptive Kernel)
# ----------------------------
class AdaKern(nn.Module):
    """
    Decompose kernel into low-frequency and high-frequency contributions.
    Implemented as:
      - lowpath: AvgPool (or blur) -> 1x1 conv (captures low-frequency)
      - highpath: input - lowpath_input -> conv (captures high-frequency)
    Then combine: out = lambda_l * lowpath_out + lambda_h * highpath_out
    Lambda predicted per-channel via global pooling + small MLP.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, reduction=4):
        super().__init__()
        padding = kernel_size // 2
        # low-frequency path: average pooling + 1x1 conv (approx mean filter -> 1x1 conv)
        self.low_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=padding, count_include_pad=False)
        self.low_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)
        # high-frequency path: residual + kxk conv
        self.high_conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=True)
        # channel-wise dynamic weights: map out_ch -> 2*out_ch then separate lambda_l and lambda_h
        mid = max(1, out_ch // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_ch, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch * 2, kernel_size=1, bias=True)  # outputs 2*out_ch values
        )
        # initialize mmlp bias such that lambdas start near (1,1)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, out_ch, H, W)
        """
        low_in = self.low_pool(x)
        low_out = self.low_conv(low_in)  # low-frequency branch
        high_in = x - low_in
        high_out = self.high_conv(high_in)  # high-frequency branch

        # predict per-channel weights from (global pooled) low+high concat (we follow paper idea: global pooling then conv)
        # choose to pool the summed branches
        pooled = self.gap(low_out + high_out)  # (B, out_ch, 1, 1)
        lambdas = self.mlp(pooled)  # (B, 2*out_ch, 1, 1)
        lambdas = lambdas.view(lambdas.shape[0], 2, -1, 1, 1)  # (B,2,out_ch,1,1)
        lambda_l = torch.sigmoid(lambdas[:, 0]) * 2.0  # scale as paper uses dynamic λ
        lambda_h = torch.sigmoid(lambdas[:, 1]) * 2.0

        out = lambda_l * low_out + lambda_h * high_out
        return out

# ----------------------------
# AdaDR (Adaptive Dilation Rate)
# ----------------------------
class AdaDR(nn.Module):
    """
    Adaptive Dilation Rate implemented as:
      - several parallel Conv2d branches each with a fixed dilation in dilation_list
      - a small conv head predicts per-pixel soft weights (softmax across branches)
      - output is per-pixel weighted sum of branch outputs
    This approximates the per-pixel dilation selection in the paper while remaining differentiable.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation_list=None, bias=True):
        super().__init__()
        if dilation_list is None:
            dilation_list = [1, 2, 4]  # typical choices; can be passed from caller
        self.dilation_list = dilation_list
        self.branches = nn.ModuleList()
        padding = kernel_size // 2
        for d in dilation_list:
            pad = (kernel_size // 2) * d
            # use padding so output spatial size preserved
            conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, dilation=d, bias=bias)
            self.branches.append(conv)
        # weight prediction head -> per-branch score map
        # predict `len(dilation_list)` channels, one per-branch, then softmax across channel dim
        self.weight_head = nn.Sequential(
            nn.Conv2d(in_ch, max(8, in_ch//2), kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, in_ch//2), len(dilation_list), kernel_size=1, bias=True)
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, out_ch, H, W)
        """
        b, c, H, W = x.shape
        scores = self.weight_head(x)  # (B, M, H, W) where M = #dilation branches
        weights = F.softmax(scores, dim=1)  # per-pixel soft selection across branches
        # compute each branch output, multiply by weight and sum
        out = 0
        for i, conv in enumerate(self.branches):
            y = conv(x)  # (B, out_ch, H, W) - because padding chosen to preserve size
            w = weights[:, i:i+1, :, :]  # (B,1,H,W)
            out = out + y * w
        return out

# ----------------------------
# Full FADC (FreqSelect -> AdaDR -> AdaKern)
# ----------------------------
class FADC(nn.Module):
    """
    Compose FreqSelect, AdaDR, AdaKern into a single replacement for Conv2d.
    Keeps simple Conv-like signature and can be directly used to replace Conv2d in many cases.
    """

    def __init__(self, in_ch=None, out_ch=None, in_channels=None, out_channels=None,
                 kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 dilation_list=None, use_freqselect=True, **kwargs):
        super().__init__()
        if in_ch is None and in_channels is not None:
            in_ch = in_channels
        if out_ch is None and out_channels is not None:
            out_ch = out_channels

        self.padding = padding
        self.stride = stride
        self.dilation = dilation

        self.freqselect = FreqSelect(in_ch) if use_freqselect else None
        self.adadr = AdaDR(in_ch, out_ch, kernel_size=kernel_size, dilation_list=dilation_list)
        # AdaKern expects the input channels; after AdaDR we have out_ch features,
        # so we set AdaKern to map out_ch -> out_ch (same channels)
        self.adakern = AdaKern(out_ch, out_ch, kernel_size=kernel_size)
        # optional BN/activation could be used outside (we keep module minimal)
        self._out_ch = out_ch

    def forward(self, x):
        # x: (B, C, H, W)
        if self.freqselect is not None:
            x_fs = self.freqselect(x)
        else:
            x_fs = x
        y = self.adadr(x_fs)
        y = self.adakern(y)
        return y

# ----------------------------
# Example quick test
# ----------------------------
if __name__ == "__main__":
    # small smoke test
    B, C, H, W = 2, 1, 32, 32
    x = torch.randn(B, C, H, W)
    mod = FADC(in_ch=1, out_ch=8, kernel_size=3, dilation_list=[1,2,4])
    y = mod(x)
    print("input:", x.shape, "output:", y.shape)

