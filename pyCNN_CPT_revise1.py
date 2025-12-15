# pyCNN修改1


import torch
import torch.nn as nn
import torch.nn.functional as F

from cftv2_head import SelectiveEnhancedCFTBlock


class pyCNN(nn.Module):
    def __init__(self, Classes, FM=64, NC=30, para_tune=True):
        super(pyCNN, self).__init__()

        # 更宽的网络，但结构更简单
        self.conv1 = nn.Sequential(
            nn.Conv2d(NC, FM * 2, 3, 1, 1),  # 增加初始通道数
            nn.BatchNorm2d(FM * 2),
            nn.ReLU(inplace=True),  # 换回ReLU
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, 3, 1, 1),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(FM * 4, FM * 8, 3, 1, 1),  # 更深的特征
            nn.BatchNorm2d(FM * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # 固定输出尺寸
            nn.Dropout(0.4),
        )

        # LiDAR分支同样加宽
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, FM * 2, 3, 1, 1),
            nn.BatchNorm2d(FM * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(FM * 2, FM * 4, 3, 1, 1),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(FM * 4, FM * 8, 3, 1, 1),
            nn.BatchNorm2d(FM * 8),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout(0.4),
        )

        # 调整CFT相关参数
        cpt_embed = FM * 8  # 匹配新的通道数
        self.proj_x1 = nn.Conv2d(FM * 8, cpt_embed, 1)
        self.proj_x2 = nn.Conv2d(FM * 8, cpt_embed, 1)
        self.ln_x1 = nn.LayerNorm(cpt_embed)

        self.cpt_block = SelectiveEnhancedCFTBlock(
            embed_dims=cpt_embed,
            num_heads=8,  # 增加注意力头
            num_classes=Classes,
            attn_drop_rate=0.1,  # 增加一点dropout
            drop_rate=0.2,
            qkv_bias=True,
            mlp_ratio=4,
            use_memory=True,  # 启用memory
            init_memory=None,
            norm_cfg=dict(type='LN', eps=1e-6),
        )

        self.fuse_beta = nn.Parameter(torch.tensor(0.3))

        # 更强大的分类头
        feature_dim = FM * 8 * 4 * 4

        self.out1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, FM * 4),
            nn.BatchNorm1d(FM * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(FM * 4, Classes)
        )

        self.out2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, FM * 4),
            nn.BatchNorm1d(FM * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(FM * 4, Classes)
        )

        self.out3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, FM * 4),
            nn.BatchNorm1d(FM * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(FM * 4, Classes)
        )

    def forward(self, x1, x2):
        # 输入处理
        if x2.dim() != 4 or x2.size(1) != 1:
            x2 = x2.unsqueeze(1)

        # HSI分支
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)

        # LiDAR分支
        x2 = self.conv4(x2)
        x2 = self.conv5(x2)
        x2 = self.conv6(x2)

        # CFT融合
        x1p = self.proj_x1(x1)
        x1p_perm = x1p.permute(0, 2, 3, 1)
        x1p_norm = self.ln_x1(x1p_perm).permute(0, 3, 1, 2)

        x2_low = F.adaptive_avg_pool2d(x2, x1p_norm.shape[2:])
        x2p = self.proj_x2(x2_low)

        outs = self.cpt_block(x1p_norm, x2p)
        cpt_feat = outs['out']

        # 融合
        fused = x1 + self.fuse_beta * cpt_feat

        # 展平
        x1_flat = x1.view(x1.size(0), -1)
        x2_flat = x2.view(x2.size(0), -1)
        fused_flat = fused.view(fused.size(0), -1)

        out1 = self.out1(x1_flat)
        out2 = self.out2(x2_flat)
        out3 = self.out3(fused_flat)

        return out1, out2, out3