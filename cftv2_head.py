import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmcv import cnn
from mmengine.model import BaseModule


class SelectiveEnhancedCFTBlock(BaseModule):
    """选择性增强的CFT块，只在关键位置进行增强"""

    def __init__(self, embed_dims, num_heads, num_classes, attn_drop_rate=.0, drop_rate=.0, qkv_bias=True,
                 mlp_ratio=4, use_memory=True, init_memory=None, norm_cfg=None, init_cfg=None,
                 enhance_prototype=True, enhance_fusion=True):
        super(SelectiveEnhancedCFTBlock, self).__init__(init_cfg)
        norm_cfg = dict(type='LN', eps=1e-6) if not norm_cfg else norm_cfg

        self.enhance_prototype = enhance_prototype
        self.enhance_fusion = enhance_fusion

        # 保持原始归一化结构
        _, self.norm_low = cnn.build_norm_layer(norm_cfg, num_features=embed_dims)
        _, self.norm_high = cnn.build_norm_layer(norm_cfg, num_features=embed_dims)


        self.cross_attn = EnhancedPrototypeCFTransform(
            embed_dims, num_heads, num_classes, attn_drop_rate,
            drop_rate, qkv_bias, use_memory=use_memory, init_memory=init_memory
        )


        # 选择性增强MLP
        if enhance_fusion:
            _, self.norm_mlp = cnn.build_norm_layer(norm_cfg, num_features=embed_dims)
            ffn_channels = embed_dims * mlp_ratio
            self.mlp = EnhancedMLP(
                embed_dims, ffn_channels, drop_rate=drop_rate
            )
        else:
            _, self.norm_mlp = cnn.build_norm_layer(norm_cfg, num_features=embed_dims)
            ffn_channels = embed_dims * mlp_ratio
            self.mlp = nn.Sequential(
                nn.Conv2d(embed_dims, ffn_channels, 1, bias=True),
                nn.Conv2d(ffn_channels, ffn_channels, 3, 1, 1, groups=ffn_channels, bias=True),
                cnn.build_activation_layer(dict(type='GELU')),
                nn.Dropout(drop_rate),
                nn.Conv2d(ffn_channels, embed_dims, 1, bias=True),
                nn.Dropout(drop_rate)
            )

    def forward(self, low, high, momentum=0.1):
        query = self.norm_low(low.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # HSI
        key_value = self.norm_high(high.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # Lidar
        outs = self.cross_attn(query, key_value, momentum)

        out = outs.pop('out') + low
        out = self.mlp(self.norm_mlp(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)) + out
        outs.update({'out': out})
        return outs


class EnhancedPrototypeCFTransform(BaseModule):
    """增强原型生成的CFTransform"""

    def __init__(self, embed_dims, num_heads, num_classes, attn_drop_rate=.0, drop_rate=.0, qkv_bias=True,
                 qk_scale=None, proj_bias=True, use_memory=True, init_memory=None, init_cfg=None):
        super(EnhancedPrototypeCFTransform, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims ** -0.5

        # 保持原始查询结构
        self.q = cnn.DepthwiseSeparableConvModule(embed_dims, embed_dims, 3, 1, 1,
                                                  act_cfg=None, bias=qkv_bias)

        # 增强的原型生成
        self.kv = EnhancedCFEmbedding(
            embed_dims, num_classes, use_memory, init_memory, qkv_bias
        )

        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Conv2d(embed_dims, embed_dims, 1, bias=proj_bias)
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, query, key_value, momentum=0.1):
        B, _, H, W = query.shape
        q = self.q(query)
        outs = self.kv(key_value, momentum)
        k, v = torch.chunk(outs.pop('out'), chunks=2, dim=1)

        q = q.reshape(B, self.num_heads, self.head_dims, -1).permute(0, 1, 3, 2)
        k = k.reshape(B, self.num_heads, self.head_dims, -1).permute(0, 1, 3, 2)
        v = v.reshape(B, self.num_heads, self.head_dims, -1).permute(0, 1, 3, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.max(attn, -1, keepdim=True)[0].expand_as(attn) - attn
        attn = F.softmax(attn, dim=-1)
        outs.update({'attn': torch.mean(attn, dim=1, keepdim=False)})
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(-2, -1).reshape(B, self.embed_dims, H, W)
        out = self.proj_drop(self.proj(out))
        outs.update({'out': out})
        return outs


class EnhancedCFEmbedding(BaseModule):
    """优化的类别原型生成，改进记忆稳定性"""

    def __init__(self, embed_dims, num_classes, use_memory, init_memory=None, kv_bias=True,
                 num_groups=4, memory_dropout=0.05, init_cfg=None):
        super(EnhancedCFEmbedding, self).__init__(init_cfg)

        self.use_memory = use_memory
        self.memory_dropout = memory_dropout

        if use_memory:
            # 改进的记忆初始化
            if init_memory is None:
                memory = torch.empty(1, num_classes, embed_dims)
                nn.init.xavier_uniform_(memory)   # 均匀初始化
            else:
                memory = torch.tensor(np.load(init_memory), dtype=torch.float)[:, :embed_dims].unsqueeze(0)

            memory = F.normalize(memory, dim=2, p=2)
            self.register_buffer('memory', memory)  # 注册缓冲区，不参与梯度下降、反向传播

            # 轻量级记忆交互
            self.memory_attention = nn.MultiheadAttention(
                embed_dims,
                num_heads=2,  # 减少头数
                batch_first=True,
                dropout=0.05,
                kdim=embed_dims,
                vdim=embed_dims
            )

            # 记忆门控
            self.memory_gate = nn.Sequential(
                nn.Linear(embed_dims * 2, embed_dims // 2),
                nn.ReLU(),
                nn.Linear(embed_dims // 2, 1),
                nn.Sigmoid()
            )

            self.memory_dropout_layer = nn.Dropout(memory_dropout)

        # 掩码学习器
        self.mask_learner = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, 3, 1, 1, groups=num_groups, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.GELU(),
            nn.Conv2d(embed_dims, num_classes, 1, bias=False)
        )

        # 特征对齐
        self.align_conv = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, 3, 1, 1, groups=num_groups, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.GELU()
        )

        # 保持原有的嵌入结构
        self.cf_embed = nn.Linear(embed_dims, embed_dims * 2, bias=kv_bias)

    @torch.no_grad()
    def _update_memory(self, cf_feat, momentum=0.1):
        """稳定的记忆更新"""
        cf_feat_mean = cf_feat.mean(dim=0, keepdim=True)
        cf_feat_mean = reduce_mean(cf_feat_mean)   # 对所有GPU上的结果求平均
        cf_feat_norm = F.normalize(cf_feat_mean, dim=2, p=2)

        # 带温度系数的动量更新
        temperature = 0.3  # 保守更新
        adjusted_momentum = momentum * temperature

        self.memory = (1.0 - adjusted_momentum) * self.memory + adjusted_momentum * cf_feat_norm
        self.memory = F.normalize(self.memory, dim=2, p=2)

    def forward(self, x, momentum=0.1):
        mask = self.mask_learner(x)
        outs = {'mask': mask}
        mask_softmax = F.softmax(mask.reshape(mask.size(0), mask.size(1), -1), dim=-1)

        x_aligned = self.align_conv(x)
        x_flat = x_aligned.reshape(x_aligned.size(0), x_aligned.size(1), -1)
        cf_feat = mask_softmax @ x_flat.transpose(-2, -1)  # (B,num_classes,embed_dims)

        if self.use_memory and hasattr(self, 'memory'):
            memory = self.memory.expand(cf_feat.size(0), -1, -1)  # 扩展维度(1->B) (B,num_classes,embed_dims)

            if self.training:
                self._update_memory(cf_feat, momentum)

                # 训练时使用dropout
                if self.memory_dropout > 0:
                    memory = self.memory_dropout_layer(memory)

            # 门控记忆交互
            attended_cf, _ = self.memory_attention(cf_feat, memory, memory)  # 自注意力机制，当前特征作为q，记忆作为kv

            gate_input = torch.cat([cf_feat, attended_cf], dim=-1)
            gate_weights = self.memory_gate(gate_input)  # 融合权重

            # 自适应融合
            cf_feat = cf_feat + gate_weights * attended_cf

        out = self.cf_embed(cf_feat)
        outs.update({'out': out.transpose(-2, -1)})
        return outs


class EnhancedMLP(BaseModule):
    """增强的MLP，增加残差连接和门控"""

    def __init__(self, embed_dims, ffn_channels, drop_rate=0.0, init_cfg=None):
        super(EnhancedMLP, self).__init__(init_cfg)

        self.conv1 = nn.Conv2d(embed_dims, ffn_channels, 1, bias=True)
        self.depthwise_conv = nn.Conv2d(ffn_channels, ffn_channels, 3, 1, 1,
                                        groups=ffn_channels, bias=True)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop_rate)

        # 门控机制
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ffn_channels, ffn_channels // 4, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(ffn_channels // 4, ffn_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.conv2 = nn.Conv2d(ffn_channels, embed_dims, 1, bias=True)
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.depthwise_conv(out)
        out = self.act(out)
        out = self.drop1(out)

        # 应用门控
        gate_weights = self.gate(out)
        out = out * gate_weights

        out = self.conv2(out)
        out = self.drop2(out)

        return out + identity


class EnhancedFusionModule(BaseModule):
    """增强的特征融合模块"""

    def __init__(self, embed_dims, num_levels, init_cfg=None):
        super(EnhancedFusionModule, self).__init__(init_cfg)

        # 注意力权重学习
        self.weight_learning = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dims, embed_dims // 4, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(embed_dims // 4, num_levels, 1, bias=False),
            nn.Softmax(dim=1)
        )

        self.fusion_conv = cnn.ConvModule(
            embed_dims * num_levels, embed_dims,
            kernel_size=3, stride=1, padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )

    def forward(self, level_features):
        # 将所有特征上采样到最大尺度
        target_size = level_features[0].shape[2:]
        upsampled_features = []

        for feat in level_features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            upsampled_features.append(feat)

        # 学习各层权重
        weights = []
        for feat in upsampled_features:
            weight = self.weight_learning(feat)
            weights.append(weight)

        # 加权融合
        weighted_sum = torch.zeros_like(upsampled_features[0])
        for i, feat in enumerate(upsampled_features):
            weight_idx = min(i, len(weights) - 1)
            weight = weights[weight_idx][:, i:i + 1] if i < len(weights) else weights[-1][:, -1:]
            weighted_sum += feat * weight

        # 最终融合
        concat_features = torch.cat(upsampled_features, dim=1)
        fused_features = self.fusion_conv(concat_features)

        return fused_features + weighted_sum  # 残差连接

def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor
