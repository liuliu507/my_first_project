
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

import math
from mmcv import cnn
from mmengine.model import BaseModule, ModuleList
from mmseg.models.builder import build_loss
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

# 从 cftv2_head.py 中导入以下类：
from cftv2_head import CFTBlock


class pyCNN(nn.Module):
    def __init__(self,Classes,FM=64,NC=30,para_tune=True):
        super(pyCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = NC,out_channels = FM,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm2d(FM),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(0.5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(FM, FM*2, 3, 1, 1),
            nn.BatchNorm2d(FM*2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )


        self.conv3 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, kernel_size=3, stride=1, padding=1),  # 普通卷积
            nn.BatchNorm2d(FM * 4),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(1, FM, 3, 1, 1, ),
            nn.BatchNorm2d(FM),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(0.5),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(FM, FM*2, 3, 1, 1),
            nn.BatchNorm2d(FM*2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, kernel_size=3, stride=1, padding=1),  # 普通卷积
            nn.BatchNorm2d(FM * 4),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
        )


        cpt_embed = FM * 4  # 或 256
        self.proj_x1 = nn.Conv2d(FM * 4, cpt_embed, 1)
        self.proj_x2 = nn.Conv2d(FM * 4, cpt_embed, 1)
        self.ln_x1 = nn.LayerNorm(cpt_embed)
        self.cpt_block = CFTBlock(embed_dims=cpt_embed,
                                  num_heads=4,
                                  num_classes=Classes,
                                  attn_drop_rate=0.0,
                                  drop_rate=0.1,
                                  qkv_bias=True,
                                  mlp_ratio=4,
                                  use_memory=False,
                                  init_memory=None,
                                  norm_cfg=dict(type='LN', eps=1e-6))
        self.fuse_beta = 0.5  # residual 权重，后续可改为 learnable nn.Parameter


        self.out1 = nn.Linear(FM * 4, Classes)
        self.out2 = nn.Linear(FM * 4, Classes)
        self.out3 = nn.Linear(FM * 4, Classes)

    def forward(self, x1, x2):

        if x1.dim() != 4 or x1.size(1) != 30:
            print(f"Warning: HSI shape mismatch! Got {x1.shape}, expected [batch,30,h,w]")
        if x2.dim() != 4 or x2.size(1) != 1:
            print(f"Warning: LiDAR shape mismatch! Got {x2.shape}, expected [batch,1,h,w]")
            x2 = x2.unsqueeze(1)  # 自动添加通道维度

        x1 = self.conv1(x1)
        x2 = self.conv4(x2)

        x1 = self.conv2(x1)
        x2 = self.conv5(x2)

        x1 = self.conv3(x1)
        x2 = self.conv6(x2)

        x1p = self.proj_x1(x1)  # [B,embed,H,W]
        # 将 x1p 进行 LN（LayerNorm 需要 channel-last）
        x1p_perm = x1p.permute(0, 2, 3, 1)  # B,H,W,C
        x1p_norm = self.ln_x1(x1p_perm).permute(0, 3, 1, 2)  # back to B,C,H,W

        # 下采样 x2
        x2_low = F.adaptive_avg_pool2d(x2, (max(1, x2.size(2) // 2), max(1, x2.size(3) // 2)))
        x2p = self.proj_x2(x2_low)

        # CPT 调用（注意参数顺序：low=high-res query, high=low-res key/value）
        outs = self.cpt_block(x1p_norm, x2p)
        cpt_feat = outs['out']  # [B, embed, H, W] (matches x1p_norm spatial)

        # residual 融合 回到分类头
        fused = x1 + cpt_feat * self.fuse_beta  # 若通道不一致，可先投回 FM*4
        fused_flat = fused.view(fused.size(0), -1)
        out3 = self.out3(fused_flat)


        x1 = x1.view(x1.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        out1 = self.out1(x1)

        x2 = x2.view(x2.size(0), -1)
        out2 = self.out2(x2)

        # # 调用 CPT 融合 HSI 和 LiDAR 特征
        # cpt_out = self.cpt_block(x1.unsqueeze(-1).unsqueeze(-1),  # low
        #                          x2.unsqueeze(-1).unsqueeze(-1))  # high
        # x = cpt_out['out'].view(x1.size(0), -1)  # 拉平成向量
        # out3 = self.out3(x)

        # x = x1 + x2
        # # x = x.view(x.size(0), 1, 2, -1)
        # out3 = self.out3(x)

        # x = torch.cat([x1, x2], dim=1)
        # x = x.view(x.size(0), 1, 2, -1)
        # out3 = self.out3(x)
        return out1, out2, out3