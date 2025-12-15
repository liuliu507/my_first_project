# pyCNN 修改2

import torch
import torch.nn as nn
import torch.nn.functional as F

from cftv2_head import SelectiveEnhancedCFTBlock


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (lightweight)."""
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        s = F.adaptive_avg_pool2d(x, 1).view(b, c)
        s = self.fc1(s)
        s = self.relu(s)
        s = self.fc2(s)
        s = self.sigmoid(s).view(b, c, 1, 1)
        return x * s


class ResConvBlock(nn.Module):
    """Residual conv block: conv - bn - relu - conv - bn + skip"""
    def __init__(self, in_ch, out_ch, downsample=False, drop=0.0):
        super(ResConvBlock, self).__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.proj = None
        if in_ch != out_ch or downsample:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.proj is not None:
            identity = self.proj(identity)
        out += identity
        out = self.relu(out)
        return out


class pyCNN(nn.Module):
    def __init__(self, Classes, FM=64, NC=30, para_tune=True):
        """
        Final integrated version:
         - keeps out1/out2/out3 outputs
         - channel-wise gating on CPT output (per-sample)
         - adaptive modality fusion weights (softmax over [h, l, cpt])
         - Dropout2d before CPT (spatial/structural regularization)
         - CPT internal drop/attn lowered for better small-sample generalization
        """
        super(pyCNN, self).__init__()

        # ---- HSI branch (keep structure) ----
        self.h1 = ResConvBlock(NC, FM * 2, downsample=False, drop=0.0)
        self.hp1_pool = nn.MaxPool2d(2)

        self.h2 = ResConvBlock(FM * 2, FM * 4, downsample=False, drop=0.2)
        self.hp2_pool = nn.MaxPool2d(2)

        self.h3 = ResConvBlock(FM * 4, FM * 8, downsample=False, drop=0.3)
        self.h3_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.h_se = SEBlock(FM * 8, reduction=8)

        # ---- LiDAR branch ----
        self.l1 = ResConvBlock(1, FM * 2, downsample=False, drop=0.0)
        self.lp1_pool = nn.MaxPool2d(2)

        self.l2 = ResConvBlock(FM * 2, FM * 4, downsample=False, drop=0.2)
        self.lp2_pool = nn.MaxPool2d(2)

        self.l3 = ResConvBlock(FM * 4, FM * 8, downsample=False, drop=0.3)
        self.l3_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.l_se = SEBlock(FM * 8, reduction=8)

        # ---- CPT prep ----
        cpt_embed = FM * 8
        self.proj_x1 = nn.Conv2d(FM * 8, cpt_embed, 1)
        self.proj_x2 = nn.Conv2d(FM * 8, cpt_embed, 1)
        self.ln_x1 = nn.LayerNorm(cpt_embed)

        # light cross-fuse before CPT
        self.cross_fuse = nn.Sequential(
            nn.Conv2d(cpt_embed * 2, cpt_embed, kernel_size=1, bias=False),
            nn.BatchNorm2d(cpt_embed),
            nn.ReLU(inplace=True)
        )

        # small structural dropout (acts like DropBlock-lite)
        self.pre_cpt_dropout = nn.Dropout2d(0.18)

        # ---- CPT block (reduced internal drop/attn for small-sample stability) ----
        self.cpt_block = SelectiveEnhancedCFTBlock(
            embed_dims=cpt_embed,
            num_heads=8,
            num_classes=Classes,
            attn_drop_rate=0.05,  # reduced
            drop_rate=0.1,        # reduced
            qkv_bias=True,
            mlp_ratio=4,
            use_memory=True,
            init_memory=None,
            norm_cfg=dict(type='LN', eps=1e-6),
        )

        # project CPT back to FM*8 and BN+ReLU
        self.cpt_proj_back = nn.Sequential(
            nn.Conv2d(cpt_embed, FM * 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(FM * 8),
            nn.ReLU(inplace=True)
        )

        # CPT output dropout (small) to avoid over-reliance
        self.cpt_dropout = nn.Dropout2d(0.12)

        # ---- channel-wise gating network: from global H+L -> per-channel gate for CPT ----
        self.gate_fc = nn.Sequential(
            nn.Linear(FM * 8 * 2, FM * 4),
            nn.ReLU(inplace=True),
            nn.Linear(FM * 4, cpt_embed),
            nn.Sigmoid()
        )

        # ---- adaptive modality fusion weights (learnable, normalized by softmax) ----
        # initial values favor HSI slightly
        self.fuse_weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))  # h, l, cpt
        self.fuse_softmax = nn.Softmax(dim=0)

        # fused BN for stability
        self.fused_bn = nn.BatchNorm2d(FM * 8)

        # ---- classification heads (keep out1/out2/out3) ----
        feature_dim = FM * 8 * 4 * 4

        def make_head():
            return nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(feature_dim, FM * 4),
                nn.BatchNorm1d(FM * 4),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(FM * 4, Classes)
            )

        self.out1 = make_head()
        self.out2 = make_head()
        self.out3 = make_head()

        # init weights
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                try:
                    if hasattr(m, 'weight') and m.weight is not None:
                        nn.init.ones_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                except Exception:
                    pass

    def forward(self, x1, x2):
        # ensure lidar has channel dim
        if x2.dim() != 4 or x2.size(1) != 1:
            x2 = x2.unsqueeze(1)

        # ---- HSI branch ----
        h = self.h1(x1)
        h = self.hp1_pool(h)
        h = self.h2(h)
        h = self.hp2_pool(h)
        h = self.h3(h)
        h = self.h3_pool(h)
        h = self.h_se(h)  # (B, FM*8, 4, 4)

        # ---- LiDAR branch ----
        l = self.l1(x2)
        l = self.lp1_pool(l)
        l = self.l2(l)
        l = self.lp2_pool(l)
        l = self.l3(l)
        l = self.l3_pool(l)
        l = self.l_se(l)  # (B, FM*8, 4, 4)

        # ---- Prepare CPT inputs ----
        x1p = self.proj_x1(h)  # (B, C, 4, 4)
        x2_low = F.adaptive_avg_pool2d(l, x1p.shape[2:])  # align spatial
        x2p = self.proj_x2(x2_low)

        # concat & light fusion -> dropout -> layernorm (channels-last) -> CPT
        cat = torch.cat([x1p, x2p], dim=1)  # (B, 2C, 4, 4)
        cat = self.cross_fuse(cat)
        cat = self.pre_cpt_dropout(cat)
        cat_perm = cat.permute(0, 2, 3, 1)  # (B, H, W, C)
        cat_norm = self.ln_x1(cat_perm).permute(0, 3, 1, 2)  # (B, C, H, W)

        outs = self.cpt_block(cat_norm, x2p)
        cpt_feat = outs['out']  # (B, cpt_embed, H, W)

        # project back to main channel dim, small dropout
        cpt_feat = self.cpt_proj_back(cpt_feat)  # (B, FM*8, H, W)
        cpt_feat = self.cpt_dropout(cpt_feat)

        # ---- compute channel-wise gate from global H+L descriptors ----
        hg = F.adaptive_avg_pool2d(h, 1).view(h.size(0), -1)  # (B, FM*8)
        lg = F.adaptive_avg_pool2d(l, 1).view(l.size(0), -1)  # (B, FM*8)
        gate_input = torch.cat([hg, lg], dim=1)  # (B, FM*8*2)
        ch_gate = self.gate_fc(gate_input)  # (B, cpt_embed) in (0,1)
        ch_gate = ch_gate.view(ch_gate.size(0), ch_gate.size(1), 1, 1)  # (B, C, 1, 1)

        # apply channel-wise gate to CPT (per-sample, per-channel)
        cpt_feat = cpt_feat * ch_gate

        # ---- adaptive modality fusion (softmax-normalized weights) ----
        w = self.fuse_softmax(self.fuse_weights)  # 3 scalars sum=1
        # ensure cpt spatial matches h/l (should already be 4x4)
        # fused is weighted sum of h, l, cpt_feat
        fused = w[0] * h + w[1] * l + w[2] * cpt_feat
        fused = self.fused_bn(fused)

        # ---- final flatten and heads (keep out1/out2/out3) ----
        h_flat = h.view(h.size(0), -1)
        l_flat = l.view(l.size(0), -1)
        fused_flat = fused.view(fused.size(0), -1)

        out1 = self.out1(h_flat)
        out2 = self.out2(l_flat)
        out3 = self.out3(fused_flat)

        return out1, out2, out3
