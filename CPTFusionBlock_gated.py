import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- 门控融合模块 ----------
class GatedFusion(nn.Module):
    """
    输入: hsi, lidar  [B, C, H, W]
    输出: fused      [B, C, H, W]
    原理：用 sigmoid 学一个 0~1 的权重 g，让网络自己决定要不要用 LiDAR
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1),  # 1×1 卷积比全连接更省
            nn.Sigmoid()
        )

    def forward(self, hsi, lidar):
        combined = torch.cat([hsi, lidar], dim=1)   # [B, 2C, H, W]
        g = self.gate(combined)                     # [B, C, H, W]  0~1
        return hsi * (1 - g) + lidar * g            # 软融合


# ---------- 带门控的 CPT 融合块 ----------
class CPTFusionBlock(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads=4, use_memory=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 交叉注意力保持不变
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # 类别原型保持不变（你可以后续再动）
        self.prototype = nn.Parameter(torch.randn(num_classes, embed_dim))
        self.use_memory = use_memory
        if use_memory:
            self.memory = nn.Parameter(torch.randn(num_classes, embed_dim))

        # FFN 保持不变
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # ✅ 新增：门控融合
        self.gated = GatedFusion(embed_dim)


    def forward(self, hsi_feat, lidar_feat, label=None):
        B, C, H, W = hsi_feat.shape

        # 1. 先门控融合一把（空间域）
        fused_map = self.gated(hsi_feat, lidar_feat)        # [B, C, H, W]

        # 2. 拉直做交叉注意力（通道域）
        hsi = fused_map.flatten(2).transpose(1, 2)          # [B, HW, C]
        lidar = lidar_feat.flatten(2).transpose(1, 2)       # LiDAR 仍做 K/V
        hsi = self.norm1(hsi)
        lidar = self.norm2(lidar)
        fused, _ = self.cross_attn(hsi, lidar, lidar)
        out = fused + hsi

        # 3. FFN
        out = out + self.ffn(out)

        # 4. 还原空间
        out = out.transpose(1, 2).view(B, C, H, W)
        return out



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


        self.out1 = nn.Linear(FM * 4, Classes)
        self.out2 = nn.Linear(FM * 4, Classes)
        self.out3 = nn.Linear(FM * 4, Classes)

        self.cpt_fusion = CPTFusionBlock(embed_dim=FM * 4, num_classes=Classes)

    def prototype_loss(self, feat, label):
        # feat: [B, C]
        # label: [B]
        with torch.no_grad():
            label_feat = self.cpt_fusion.prototype[label]  # 从融合模块里拿原型
        loss = F.mse_loss(F.normalize(feat, dim=1),
                          F.normalize(label_feat.detach(), dim=1))
        return loss * 0.1

    def forward(self, x1, x2,labels,return_mask=False):

        if x1.dim() != 4 or x1.size(1) != 30:
            print(f"Warning: HSI shape mismatch! Got {x1.shape}, expected [batch,30,h,w]")
        if x2.dim() != 4 or x2.size(1) != 1:
            print(f"Warning: LiDAR shape mismatch! Got {x2.shape}, expected [batch,1,h,w]")
            x2 = x2.unsqueeze(1)  # 自动添加通道维度

        x1 = self.conv1(x1)
        x2 = self.conv4(x2)

        x1 = self.conv2(x1)
        x2 = self.conv5(x2)

        x1 = self.conv3(x1)  # [B, FM*4, H, W]
        x2 = self.conv6(x2)  # [B, FM*4, H, W]

        # ✅ CPT 融合（保持空间维度）
        fused = self.cpt_fusion(x1, x2)  # [B, FM*4, H, W]

        feat_global = F.adaptive_avg_pool2d(fused, (1, 1)).squeeze(-1).squeeze(-1)  # [B, FM*4]

        # ✅ 展平并送入分类器
        x = fused.view(fused.size(0), -1)
        out3 = self.out3(x)

        # 单模态输出保留（可选）
        x1_flat = x1.view(x1.size(0), -1)
        x2_flat = x2.view(x2.size(0), -1)
        out1 = self.out1(x1_flat)
        out2 = self.out2(x2_flat)

        proto_loss = 0
        if labels is not None:
            proto_loss = self.prototype_loss(feat_global, labels)

        return out1, out2, out3, proto_loss
