import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
import math
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
            nn.Dropout(0.6),
        )


        self.conv3 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, kernel_size=3, stride=1, padding=1),  # 普通卷积
            nn.BatchNorm2d(FM * 4),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.6),
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
            nn.Dropout(0.6),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, kernel_size=3, stride=1, padding=1),  # 普通卷积
            nn.BatchNorm2d(FM * 4),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.6),
        )


        cpt_embed = FM * 4
        self.proj_x1 = nn.Conv2d(FM * 4, cpt_embed, 1)
        self.proj_x2 = nn.Conv2d(FM * 4, cpt_embed, 1)
        self.ln_x1 = nn.LayerNorm(cpt_embed)
        self.cpt_block = CFTBlock(embed_dims=cpt_embed,
                                  num_heads=4,
                                  num_classes=Classes,
                                  attn_drop_rate=0.05,
                                  drop_rate=0.1,
                                  qkv_bias=True,
                                  mlp_ratio=2,
                                  use_memory=True,
                                  init_memory=None,
                                  norm_cfg=dict(type='LN', eps=1e-6))
        self.fuse_beta = 0.5
        # self.fuse_beta = nn.Parameter(torch.tensor(0.3))

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

        # 改为可学习的加权融合
        if hasattr(self, 'fuse_beta') and isinstance(self.fuse_beta, nn.Parameter):
            beta = torch.sigmoid(self.fuse_beta)  # 确保在0-1之间
        else:
            beta = self.fuse_beta

        # 去掉门控机制，直接进行残差融合（消融实验）
        fused = x1 + cpt_feat * beta
        # ================================

        fused_flat = fused.view(fused.size(0), -1)
        out3 = self.out3(fused_flat)

        x1 = x1.view(x1.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        out1 = self.out1(x1)

        x2 = x2.view(x2.size(0), -1)
        out2 = self.out2(x2)

        return out1, out2, out3
