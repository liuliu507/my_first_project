import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSpatialGate(nn.Module):
    """
    通道-空间联合门控（Channel-Spatial Joint Gate）
    
    结合通道注意力和空间注意力，生成联合门控权重。
    """

    def __init__(self, channels):
        super(ChannelSpatialGate, self).__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # 通道注意力
        channel_gate1 = self.channel_attn(x1)
        channel_gate2 = self.channel_attn(x2)
        channel_gate = (channel_gate1 + channel_gate2) / 2

        # 空间注意力
        avg_pool = torch.cat([torch.mean(x1, dim=1, keepdim=True),
                              torch.mean(x2, dim=1, keepdim=True)], dim=1)
        spatial_gate = self.spatial_attn(avg_pool)

        # 联合门控
        joint_gate = channel_gate * spatial_gate

        return joint_gate


class TaskGuidedFusionV2(nn.Module):
    """
    任务驱动的特征融合修正V2（完整实现版）
    
    封装完整的任务反馈逻辑，包括：
    1. 通道-空间联合门控
    2. 可学习融合权重
    3. 任务梯度计算
    4. 错误样本识别与修正
    
    Args:
        embed_dim: 特征维度
        step_size: 梯度修正步长
    """

    def __init__(self, embed_dim, step_size=0.1):
        super(TaskGuidedFusionV2, self).__init__()
        self.embed_dim = embed_dim
        self.step_size = step_size
        
        # 可学习融合权重
        self.fuse_beta = nn.Parameter(torch.tensor(0.5))
        
        # 通道-空间联合门控
        self.gate = ChannelSpatialGate(embed_dim)

    def compute_joint_gate(self, hsi_feat, cpt_feat):
        """
        计算通道-空间联合门控
        
        Args:
            hsi_feat: HSI特征 [B, C, H, W]
            cpt_feat: CPT特征 [B, C, H, W]
        
        Returns:
            joint_gate: 联合门控权重 [B, C, H, W]
            cpt_weighted: 加权后的CPT特征 [B, C, H, W]
        """
        # 1. 通道注意力：捕捉类别信息
        channel_attn_hsi = torch.sigmoid(torch.mean(hsi_feat, dim=(2, 3), keepdim=True))
        channel_attn_cpt = torch.sigmoid(torch.mean(cpt_feat, dim=(2, 3), keepdim=True))
        channel_gate = (channel_attn_hsi + channel_attn_cpt) / 2

        # 2. 空间注意力：捕捉位置信息
        spatial_attn_hsi = torch.sigmoid(torch.mean(hsi_feat, dim=1, keepdim=True))
        spatial_attn_cpt = torch.sigmoid(torch.mean(cpt_feat, dim=1, keepdim=True))
        spatial_gate = (spatial_attn_hsi + spatial_attn_cpt) / 2

        # 3. 联合门控：通道和空间信息的联合
        joint_gate = channel_gate * spatial_gate
        
        # 4. 特征加权
        cpt_weighted = cpt_feat * joint_gate
        
        return joint_gate, cpt_weighted

    def compute_task_gradient(self, fused_feat, target, classifier):
        """
        计算任务梯度（核心函数）
        
        Args:
            fused_feat: 融合特征 [B, C, H, W]
            target: 目标标签 [B]
            classifier: 主分类器（用于预测）
        
        Returns:
            task_gradient: 任务梯度 [B, C, H, W] 或 None
            is_wrong: 错误样本掩码 [B]
        """
        # 分离特征，创建独立的计算图（不破坏主梯度流）
        fused_for_task = fused_feat.detach().requires_grad_(True)
        
        # 使用主分类器进行预测
        fused_flat = fused_for_task.view(fused_for_task.size(0), -1)
        out = classifier(fused_flat)
        
        # 计算预测是否正确
        pred_labels = out.argmax(dim=1)
        is_wrong = (pred_labels != target).float()  # 1表示预测错误，0表示正确
        
        # 如果没有错误样本，直接返回
        if is_wrong.sum() == 0:
            return None, is_wrong
        
        # 计算任务损失（只关注错误样本）
        task_loss = F.cross_entropy(out, target, reduction='none')
        weighted_task_loss = (task_loss * is_wrong).sum() / (is_wrong.sum() + 1e-10)
        
        # 反向传播获取梯度
        weighted_task_loss.backward(retain_graph=True)
        
        # 获取任务梯度
        task_gradient = fused_for_task.grad
        
        return task_gradient, is_wrong

    def normalize_gradient(self, gradient):
        """
        L2归一化梯度
        
        Args:
            gradient: 原始梯度 [B, C, H, W]
        
        Returns:
            normalized_gradient: 归一化后的梯度 [B, C, H, W]
        """
        if gradient is None:
            return None
        
        grad_norm = torch.norm(gradient.view(gradient.size(0), -1), dim=1, keepdim=True)
        grad_norm = grad_norm.view(gradient.size(0), 1, 1, 1)
        normalized_gradient = gradient / (grad_norm + 1e-10)
        
        return normalized_gradient

    def rectify_fusion(self, fused_initial, task_gradient, is_wrong):
        """
        应用梯度修正（只对错误样本）
        
        Args:
            fused_initial: 初始融合特征 [B, C, H, W]
            task_gradient: 归一化后的任务梯度 [B, C, H, W]
            is_wrong: 错误样本掩码 [B]
        
        Returns:
            fused: 修正后的特征 [B, C, H, W]
        """
        if task_gradient is None:
            return fused_initial
        
        # 扩展错误掩码到与梯度相同的维度
        is_wrong_expanded = is_wrong.view(-1, 1, 1, 1).expand_as(task_gradient)
        
        # 应用梯度修正：fused = initial - step_size * gradient * mask
        fused = fused_initial - self.step_size * task_gradient * is_wrong_expanded
        
        # 限制范围
        fused = torch.clamp(fused, -5, 5)
        
        return fused

    def forward(self, hsi_feat, cpt_feat, target=None, classifier=None):
        """
        前向传播（封装完整逻辑）
        
        Args:
            hsi_feat: HSI特征 [B, C, H, W]
            cpt_feat: CPT特征 [B, C, H, W]
            target: 目标标签 [B] (仅在训练时需要)
            classifier: 主分类器 (仅在训练时需要)
        
        Returns:
            fused: 融合后的特征 [B, C, H, W]
        """
        # 1. 计算联合门控和加权CPT特征
        joint_gate, cpt_weighted = self.compute_joint_gate(hsi_feat, cpt_feat)
        
        # 2. 可学习融合
        beta = torch.sigmoid(self.fuse_beta)
        fused = hsi_feat + cpt_weighted * beta
        
        # 3. 训练时进行任务驱动的修正
        if self.training and target is not None and classifier is not None:
            # 计算任务梯度
            task_gradient, is_wrong = self.compute_task_gradient(fused, target, classifier)
            
            # 如果有错误样本，进行修正
            if task_gradient is not None:
                # 归一化梯度
                task_gradient = self.normalize_gradient(task_gradient)
                # 应用修正
                fused = self.rectify_fusion(fused, task_gradient, is_wrong)
        
        return fused
