import torch
import torch.nn as nn
import torch.nn.functional as F
from cftv2_head import CFTBlock
from task_guided_fusion import TaskGuidedFusionV2


class pyCNN(nn.Module):
    def __init__(self, Classes, FM=64, NC=30, para_tune=True, use_task_guided_fusion=True):
        super(pyCNN, self).__init__()

        self.use_task_guided_fusion = use_task_guided_fusion

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=NC, out_channels=FM, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(FM),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(FM, FM * 2, 3, 1, 1),
            nn.BatchNorm2d(FM * 2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.6),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, kernel_size=3, stride=1, padding=1),
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
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(FM, FM * 2, 3, 1, 1),
            nn.BatchNorm2d(FM * 2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.6),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, kernel_size=3, stride=1, padding=1),
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

        # ========================================
        # 创新点2：任务驱动的特征融合修正（借鉴CRFM思想）
        # ========================================
        if self.use_task_guided_fusion:
            # 使用V2版本（封装完整逻辑）
            self.task_guided_fusion = TaskGuidedFusionV2(
                embed_dim=FM * 4,
                step_size=0.1
            )
        else:
            # 保留原有的简单融合参数
            self.fuse_beta = 0.5
        # ========================================

        self.out1 = nn.Linear(FM * 4, Classes)
        self.out2 = nn.Linear(FM * 4, Classes)
        self.out3 = nn.Linear(FM * 4, Classes)

    def forward(self, x1, x2, target=None):
        """
        Args:
            x1: HSI输入
            x2: LiDAR输入
            target: 目标标签（仅在训练且使用task_guided_fusion时需要）
        """
        if x1.dim() != 4 or x1.size(1) != 30:
            print(f"Warning: HSI shape mismatch! Got {x1.shape}, expected [batch,30,h,w]")
        if x2.dim() != 4 or x2.size(1) != 1:
            print(f"Warning: LiDAR shape mismatch! Got {x2.shape}, expected [batch,1,h,w]")
            x2 = x2.unsqueeze(1)

        x1 = self.conv1(x1)
        x2 = self.conv4(x2)

        x1 = self.conv2(x1)
        x2 = self.conv5(x2)

        x1 = self.conv3(x1)
        x2 = self.conv6(x2)

        x1p = self.proj_x1(x1)
        x1p_perm = x1p.permute(0, 2, 3, 1)
        x1p_norm = self.ln_x1(x1p_perm).permute(0, 3, 1, 2)

        x2_low = F.adaptive_avg_pool2d(x2, (max(1, x2.size(2) // 2), max(1, x2.size(3) // 2)))
        x2p = self.proj_x2(x2_low)

        outs = self.cpt_block(x1p_norm, x2p)
        cpt_feat = outs['out']

        # ========================================
        # 创新点2：任务驱动的特征融合修正
        # ========================================
        if self.use_task_guided_fusion:
            # 调用封装好的融合模块（包含门控、融合、任务修正的完整逻辑）
            fused = self.task_guided_fusion(
                hsi_feat=x1,
                cpt_feat=cpt_feat,
                target=target,
                classifier=self.out3
            )
        else:
            # Baseline：使用固定beta的简单融合
            beta = self.fuse_beta if hasattr(self, 'fuse_beta') else 0.5
            fused = x1 + cpt_feat * beta
        # ========================================

        fused_flat = fused.view(fused.size(0), -1)
        out3 = self.out3(fused_flat)

        x1 = x1.view(x1.size(0), -1)
        out1 = self.out1(x1)

        x2 = x2.view(x2.size(0), -1)
        out2 = self.out2(x2)

        return out1, out2, out3


# 保留原有的pyCNN类作为baseline对比
class pyCNN_baseline(nn.Module):
    """原始版本（用于对比实验）"""

    def __init__(self, Classes, FM=64, NC=30, para_tune=True):
        super(pyCNN_baseline, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=NC, out_channels=FM, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(FM),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(FM, FM * 2, 3, 1, 1),
            nn.BatchNorm2d(FM * 2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.6),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, kernel_size=3, stride=1, padding=1),
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
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(FM, FM * 2, 3, 1, 1),
            nn.BatchNorm2d(FM * 2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.6),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, kernel_size=3, stride=1, padding=1),
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

        self.out1 = nn.Linear(FM * 4, Classes)
        self.out2 = nn.Linear(FM * 4, Classes)
        self.out3 = nn.Linear(FM * 4, Classes)

    def forward(self, x1, x2):
        if x1.dim() != 4 or x1.size(1) != 30:
            print(f"Warning: HSI shape mismatch! Got {x1.shape}, expected [batch,30,h,w]")
        if x2.dim() != 4 or x2.size(1) != 1:
            print(f"Warning: LiDAR shape mismatch! Got {x2.shape}, expected [batch,1,h,w]")
            x2 = x2.unsqueeze(1)

        x1 = self.conv1(x1)
        x2 = self.conv4(x2)

        x1 = self.conv2(x1)
        x2 = self.conv5(x2)

        x1 = self.conv3(x1)
        x2 = self.conv6(x2)

        x1p = self.proj_x1(x1)
        x1p_perm = x1p.permute(0, 2, 3, 1)
        x1p_norm = self.ln_x1(x1p_perm).permute(0, 3, 1, 2)

        x2_low = F.adaptive_avg_pool2d(x2, (max(1, x2.size(2) // 2), max(1, x2.size(3) // 2)))
        x2p = self.proj_x2(x2_low)

        outs = self.cpt_block(x1p_norm, x2p)
        cpt_feat = outs['out']

        # 原有的通道-空间联合门控
        channel_attn_x1 = torch.sigmoid(torch.mean(x1, dim=(2, 3), keepdim=True))
        channel_attn_cpt = torch.sigmoid(torch.mean(cpt_feat, dim=(2, 3), keepdim=True))
        channel_gate = (channel_attn_x1 + channel_attn_cpt) / 2

        spatial_attn_x1 = torch.sigmoid(torch.mean(x1, dim=1, keepdim=True))
        spatial_attn_cpt = torch.sigmoid(torch.mean(cpt_feat, dim=1, keepdim=True))
        spatial_gate = (spatial_attn_x1 + spatial_attn_cpt) / 2

        joint_gate = channel_gate * spatial_gate
        cpt_feat_weighted = cpt_feat * joint_gate

        fused = x1 + cpt_feat_weighted * self.fuse_beta

        fused_flat = fused.view(fused.size(0), -1)
        out3 = self.out3(fused_flat)

        x1 = x1.view(x1.size(0), -1)
        out1 = self.out1(x1)

        x2 = x2.view(x2.size(0), -1)
        out2 = self.out2(x2)

        return out1, out2, out3
