import os
from operator import truediv
import torch.nn.functional as F
import numpy as np
import random
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dataf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
import scipy.io as sio
from scipy.io import savemat
import json
from sklearn.decomposition import PCA
from pymodel_fuse_updateGate import pyCNN
# from pymodel_base import pyCNN

# ==================== 设置实验参数 ====================
DATASET_NAME = "Houston"  # 数据集名称: "Trento", "Muufl", "Houston", "Augsburg"
TRAIN_NUM = 20  # 每类训练样本数
BATCH_SIZE_TRAIN = 64  # 批次大小
EPOCH = 300  # 训练轮数
LR = 0.001  # 学习率
FM = 64  # 模型特征维度

# 根据数据集设置类别数
if DATASET_NAME == "Trento":
    CLASSES_NUM = 6
elif DATASET_NAME == "Muufl":
    CLASSES_NUM = 11
elif DATASET_NAME == "Houston":
    CLASSES_NUM = 15
elif DATASET_NAME == "Augsburg":
    CLASSES_NUM = 13
else:
    raise ValueError(f"不支持的数据库: {DATASET_NAME}")

# 图像块大小
patchsize1 = 9
patchsize2 = 9
half_patch1 = patchsize1 // 2  # 以目标像素为中心的领域范围
half_patch2 = patchsize2 // 2

# 创建结果目录
RESULTS_DIR = "./experimentResults"
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_experiment_folder(dataset_name, train_num):
    """为当前实验创建文件夹"""
    folder_name = f"38_MS2CA_CPT_HSI与LIDAR融合后类别原型_更新门控融合_任务反馈_{dataset_name}"
    folder_path = os.path.join(RESULTS_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def AA_andEachClassAccuracy(confusion_matrix):
    """计算每类准确率和平均准确率"""
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test, dataset=DATASET_NAME):
    """计算评估指标"""
    if dataset == 'Trento':
        all_names = ['Apple Trees', 'Buildings', 'Ground',
                     'Woods', 'Vineyard', 'Roads']
    elif dataset == 'Muufl':
        all_names = ['Trees', 'Mostly grass', 'Mixed ground surface',
                     'Dirt and sand', 'Road', 'Water', 'Building Shadow',
                     'Building', 'Sidewalk', 'Yellow curb', 'Cloth panels']
    elif dataset == 'Houston':
        all_names = ['Healthy Grass', 'Stressed Grass', 'Synthetic Grass', 'Tree',
                     'Soil', 'Water', 'Residential', 'Commercial', 'Road', 'Highway',
                     'Railway', 'Parking Lot1', 'Parking Lot2', 'Tennis Court',
                     'Running Track']
    elif dataset == 'Augsburg':
        all_names = ['Forest', 'Residential Area', 'Industrial Area',
                     'Low Plants', 'Soil', 'Allotment',
                     'Commercial Area', 'Water', 'Railway',
                     'Harbor', 'Pasture', 'Roads', 'Urban Green']
    else:
        all_names = [f'Class {i + 1}' for i in range(max(y_test) + 1)]

    unique_labels = sorted(np.unique(y_test))
    used_names = [all_names[i] for i in unique_labels]  # 实际出现的类别标签

    classification = classification_report(
        y_test,
        y_pred_test,
        labels=unique_labels,
        digits=4,  # 浮点数的精度
        target_names=used_names
    )

    confusion = confusion_matrix(y_test, y_pred_test, labels=unique_labels)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    oa = accuracy_score(y_test, y_pred_test)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100, unique_labels


def load_raw_data(dataset_name):
    """加载原始HSI和LiDAR数据"""
    if dataset_name == "Trento":
        Data = sio.loadmat(r'./Data/Trento_hsi.mat')['HSI']
        Data2 = sio.loadmat(r'./Data/Trento_LiDAR.mat')['LiDAR']
    elif dataset_name == "Muufl":
        Data = sio.loadmat(r'./Data/Muufl_hsi.mat')['hsi']
        Data2 = sio.loadmat(r'./Data/Muufl_Lidar.mat')['lidar']
    elif dataset_name == "Houston":
        Data = sio.loadmat(r'./Data/Houston.mat')['img']
        Data2 = sio.loadmat(r'./Data/Houston_LiDAR.mat')['img']
    elif dataset_name == "Augsburg":
        Data = sio.loadmat(r'./Data/augsburg_hsi.mat')['augsburg_hsi']
        Data2 = sio.loadmat(r'./Data/augsburg_sar.mat')['augsburg_sar']
    else:
        raise ValueError(f"不支持的数据库: {dataset_name}")

    Data = Data.astype(np.float32)
    Data2 = Data2.astype(np.float32)

    # 处理LiDAR数据维度
    if len(Data2.shape) == 3 and Data2.shape[2] >= 2:
        Data2 = np.mean(Data2, axis=2)
    elif len(Data2.shape) == 3 and Data2.shape[2] == 1:
        Data2 = Data2[:, :, 0]

    return Data, Data2


def load_mask_data(dataset_name):
    """加载原始标签数据"""
    if dataset_name == "Trento":
        AllLabel = sio.loadmat(r'./Data/Trento_allgrd.mat')['mask_test']
    elif dataset_name == "Muufl":
        AllLabel = sio.loadmat(r'./Data/Muufl_gt.mat')['Muufl_gt']
    elif dataset_name == "Houston":
        AllLabel = sio.loadmat(r'./Data/Houston_gt.mat')['Houston_gt']
    elif dataset_name == "Augsburg":
        AllLabel = sio.loadmat(r'./Data/augsburg_gt.mat')['augsburg_gt']
    else:
        raise ValueError(f"不支持的数据库: {dataset_name}")

    return AllLabel


def nor_pca(Data, Data2, ispca=True):
    """
    数据归一化和PCA降维
    """
    [m, n, l] = Data.shape

    # HSI数据归一化
    for i in range(l):
        minimal = Data[:, :, i].min()
        maximal = Data[:, :, i].max()
        if maximal - minimal != 0:
            Data[:, :, i] = (Data[:, :, i] - minimal) / (maximal - minimal)
        else:
            Data[:, :, i] = 0.0

    # LiDAR数据归一化
    minimal = Data2.min()
    maximal = Data2.max()
    if maximal - minimal != 0:
        Data2 = (Data2 - minimal) / (maximal - minimal)
    else:
        Data2 = np.zeros_like(Data2)

    # PCA降维
    if ispca:
        NC = 30
        PC = np.reshape(Data, (m * n, l))
        pca = PCA(n_components=NC, copy=True, whiten=False)
        PC = pca.fit_transform(PC)
        PC = np.reshape(PC, (m, n, NC))
    else:
        NC = l
        PC = Data

    return PC, Data2, NC


def zero_pad_patch(local_patch, target_size, center_r, center_c, original_r, original_c):
    """
    对局部块进行零填充
    local_patch: 提取的局部块
    target_size: 目标尺寸 (h, w)
    center_r, center_c: 中心点在原始图像中的位置
    original_r, original_c: 局部块在原始图像中的起始位置
    """
    target_h, target_w = target_size
    actual_h, actual_w = local_patch.shape[:2]

    # 创建目标大小的零张量
    if len(local_patch.shape) == 3:
        target_patch = np.zeros((target_h, target_w, local_patch.shape[2]), dtype=local_patch.dtype)
    else:
        target_patch = np.zeros((target_h, target_w), dtype=local_patch.dtype)

    # 计算偏移量
    half_h = target_h // 2
    half_w = target_w // 2
    r_offset = half_h - (center_r - original_r)
    c_offset = half_w - (center_c - original_c)

    # 确保偏移量非负
    r_start_target = max(0, r_offset)
    c_start_target = max(0, c_offset)

    # 计算源数据的范围
    r_start_source = max(0, -r_offset)
    c_start_source = max(0, -c_offset)

    # 计算复制的尺寸
    copy_h = min(actual_h - r_start_source, target_h - r_start_target)
    copy_w = min(actual_w - c_start_source, target_w - c_start_target)

    # 复制数据
    if len(local_patch.shape) == 3:
        target_patch[r_start_target:r_start_target + copy_h,
        c_start_target:c_start_target + copy_w, :] = \
            local_patch[r_start_source:r_start_source + copy_h,
            c_start_source:c_start_source + copy_w, :]
    else:
        target_patch[r_start_target:r_start_target + copy_h,
        c_start_target:c_start_target + copy_w] = \
            local_patch[r_start_source:r_start_source + copy_h,
            c_start_source:c_start_source + copy_w]

    return target_patch


def con_data(PC, Data2_norm, TrLabel, TsLabel, NC):
    """
    构造训练和测试数据块 - 无边界填充版本
    """
    [m, n, _] = PC.shape  # 原始图像尺寸

    # ==================== 提取HSI训练数据块 ====================
    [ind1, ind2] = np.where(TrLabel != 0)
    TrainNum = len(ind1)
    TrainPatch = np.empty((TrainNum, NC, patchsize1, patchsize1), dtype='float32')
    TrainLabel = np.empty(TrainNum)

    for i in range(TrainNum):
        r, c = ind1[i], ind2[i]

        # 计算提取范围（考虑边界）
        r_start = max(0, r - half_patch1)
        r_end = min(m, r + half_patch1 + 1)
        c_start = max(0, c - half_patch1)
        c_end = min(n, c + half_patch1 + 1)

        # 提取局部块
        local_patch = PC[r_start:r_end, c_start:c_end, :]

        # 如果块大小不够，用零填充
        if local_patch.shape[0] < patchsize1 or local_patch.shape[1] < patchsize1:
            local_patch = zero_pad_patch(local_patch, (patchsize1, patchsize1), r, c, r_start, c_start)

        # 调整维度顺序
        patch = np.reshape(local_patch, (patchsize1 * patchsize1, NC))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (NC, patchsize1, patchsize1))
        TrainPatch[i, :, :, :] = patch
        TrainLabel[i] = TrLabel[r, c]

    # ==================== 提取HSI测试数据块 ====================
    [ind1, ind2] = np.where(TsLabel != 0)
    TestNum = len(ind1)
    TestPatch = np.empty((TestNum, NC, patchsize1, patchsize1), dtype='float32')
    TestLabel = np.empty(TestNum)

    for i in range(TestNum):
        r, c = ind1[i], ind2[i]

        # 计算提取范围
        r_start = max(0, r - half_patch1)
        r_end = min(m, r + half_patch1 + 1)
        c_start = max(0, c - half_patch1)
        c_end = min(n, c + half_patch1 + 1)

        # 提取局部块
        local_patch = PC[r_start:r_end, c_start:c_end, :]

        # 如果块大小不够，用零填充
        if local_patch.shape[0] < patchsize1 or local_patch.shape[1] < patchsize1:
            local_patch = zero_pad_patch(local_patch, (patchsize1, patchsize1), r, c, r_start, c_start)

        patch = np.reshape(local_patch, (patchsize1 * patchsize1, NC))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (NC, patchsize1, patchsize1))
        TestPatch[i, :, :, :] = patch
        TestLabel[i] = TsLabel[r, c]

    # ==================== 提取LiDAR训练数据块 ====================
    [ind1, ind2] = np.where(TrLabel != 0)
    TrainNum2 = len(ind1)
    TrainPatch2 = np.empty((TrainNum2, 1, patchsize2, patchsize2), dtype='float32')
    TrainLabel2 = np.empty(TrainNum2)

    for i in range(TrainNum2):
        r, c = ind1[i], ind2[i]

        # 计算提取范围
        r_start = max(0, r - half_patch2)
        r_end = min(m, r + half_patch2 + 1)
        c_start = max(0, c - half_patch2)
        c_end = min(n, c + half_patch2 + 1)

        # 提取局部块
        if len(Data2_norm.shape) == 2:
            local_patch = Data2_norm[r_start:r_end, c_start:c_end]
            # 添加通道维度
            local_patch = local_patch[:, :, np.newaxis]
        else:
            local_patch = Data2_norm[r_start:r_end, c_start:c_end, :]

        # 如果块大小不够，用零填充
        if local_patch.shape[0] < patchsize2 or local_patch.shape[1] < patchsize2:
            local_patch = zero_pad_patch(local_patch, (patchsize2, patchsize2), r, c, r_start, c_start)

        patch = np.reshape(local_patch, (patchsize2 * patchsize2, 1))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (1, patchsize2, patchsize2))
        TrainPatch2[i, :, :, :] = patch
        TrainLabel2[i] = TrLabel[r, c]

    # ==================== 提取LiDAR测试数据块 ====================
    [ind1, ind2] = np.where(TsLabel != 0)
    TestNum2 = len(ind1)
    TestPatch2 = np.empty((TestNum2, 1, patchsize2, patchsize2), dtype='float32')
    TestLabel2 = np.empty(TestNum2)

    for i in range(TestNum2):
        r, c = ind1[i], ind2[i]

        # 计算提取范围
        r_start = max(0, r - half_patch2)
        r_end = min(m, r + half_patch2 + 1)
        c_start = max(0, c - half_patch2)
        c_end = min(n, c + half_patch2 + 1)

        # 提取局部块
        if len(Data2_norm.shape) == 2:
            local_patch = Data2_norm[r_start:r_end, c_start:c_end]
            local_patch = local_patch[:, :, np.newaxis]
        else:
            local_patch = Data2_norm[r_start:r_end, c_start:c_end, :]

        # 如果块大小不够，用零填充
        if local_patch.shape[0] < patchsize2 or local_patch.shape[1] < patchsize2:
            local_patch = zero_pad_patch(local_patch, (patchsize2, patchsize2), r, c, r_start, c_start)

        patch = np.reshape(local_patch, (patchsize2 * patchsize2, 1))
        patch = np.transpose(patch)
        patch = np.reshape(patch, (1, patchsize2, patchsize2))
        TestPatch2[i, :, :, :] = patch
        TestLabel2[i] = TsLabel[r, c]

    return TrainPatch, TestPatch, TrainPatch2, TestPatch2, TrainLabel, TestLabel, TrainLabel2, TestLabel2


def create_masks_from_mat(seed, dataset_name=DATASET_NAME):
    """
    从mat文件加载训练和测试掩码
    使用data_prepare.py生成的文件: ./Results/{dataset_name}/train_test_gt_{seed}.mat
    """
    mat_path = f'./Results/{dataset_name}/train_test_gt_{seed}.mat'

    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"索引文件不存在: {mat_path}\n请先运行data_prepare.py生成索引文件")

    data = sio.loadmat(mat_path)
    train_mask = data['train_data']
    test_mask = data['test_data']

    return train_mask, test_mask


def create_data_loader(dataset_name=DATASET_NAME, train_num=TRAIN_NUM, seed=42):
    """
    创建数据加载器 - 无边界填充版本
    """
    print(f"\n使用种子 {seed} 加载数据...")

    # 1. 加载原始HSI和LiDAR数据
    Data, Data2 = load_raw_data(dataset_name)

    # 2. 加载原始标签数据（用于验证）
    AllLabel = load_mask_data(dataset_name)

    # 3. 从mat文件加载训练和测试掩码
    train_mask, test_mask = create_masks_from_mat(seed, dataset_name)

    # 验证掩码形状是否匹配
    if train_mask.shape != AllLabel.shape:
        print(f"警告: 训练掩码形状 {train_mask.shape} 与标签形状 {AllLabel.shape} 不匹配，正在调整...")
        if train_mask.shape[0] * train_mask.shape[1] == AllLabel.shape[0] * AllLabel.shape[1]:
            train_mask = train_mask.reshape(AllLabel.shape)
            test_mask = test_mask.reshape(AllLabel.shape)

    print(f"训练样本数: {np.sum(train_mask > 0)}")
    print(f"测试样本数: {np.sum(test_mask > 0)}")

    # 统计训练集类别分布
    train_unique, train_counts = np.unique(train_mask[train_mask > 0], return_counts=True)
    print("训练集类别分布:")
    for cls, count in zip(train_unique, train_counts):
        print(f"  类别 {int(cls)}: {count} 个样本")

    # 4. 归一化和PCA（保持不变）
    PC, Data2_norm, NC = nor_pca(Data, Data2, ispca=True)

    # 5. 直接构造数据块（跳过边界填充）
    TrainPatch, TestPatch, TrainPatch2, TestPatch2, TrainLabel, TestLabel, TrainLabel2, TestLabel2 = con_data(
        PC, Data2_norm, train_mask, test_mask, NC
    )

    print(f'HSI训练集大小: {TrainPatch.shape}, 测试集大小: {TestPatch.shape}')
    print(f'LiDAR训练集大小: {TrainPatch2.shape}, 测试集大小: {TestPatch2.shape}')

    # PyTorch格式转换
    TrainPatch1 = torch.from_numpy(TrainPatch).float()
    TestPatch1 = torch.from_numpy(TestPatch).float()
    TrainPatch2_tensor = torch.from_numpy(TrainPatch2).float()
    TestPatch2_tensor = torch.from_numpy(TestPatch2).float()

    # 标签从1开始，需要减1调整为从0开始
    TrainLabel1 = torch.from_numpy(TrainLabel).long() - 1
    TestLabel1 = torch.from_numpy(TestLabel).long() - 1

    print(f"训练标签范围: {TrainLabel1.min()} ~ {TrainLabel1.max()}")
    print(f"测试标签范围: {TestLabel1.min()} ~ {TestLabel1.max()}")

    # 创建训练数据集
    class TrainDS(torch.utils.data.Dataset):
        def __init__(self, hsi_data, lidar_data, labels):
            self.len = labels.shape[0]
            self.x_hsi = hsi_data
            self.x_lidar = lidar_data
            self.y = labels

        def __getitem__(self, index):
            return self.x_hsi[index], self.x_lidar[index], self.y[index]

        def __len__(self):
            return self.len

    class TestDS(torch.utils.data.Dataset):
        def __init__(self, hsi_data, lidar_data, labels):
            self.len = labels.shape[0]
            self.x_hsi = hsi_data
            self.x_lidar = lidar_data
            self.y_data = labels

        def __getitem__(self, index):
            return self.x_hsi[index], self.x_lidar[index], self.y_data[index]

        def __len__(self):
            return self.len

    trainset = TrainDS(TrainPatch1, TrainPatch2_tensor, TrainLabel1)
    testset = TestDS(TestPatch1, TestPatch2_tensor, TestLabel1)

    return trainset, testset, TrainLabel1, TestLabel1, NC


def train(train_loader, epochs, Classes, NC, device, use_task_guided_fusion=True):
    """训练模型"""
    # 初始化模型
    para_tune = True
    # 学生模型
    cnn = pyCNN(FM=FM, NC=NC, Classes=Classes, para_tune=para_tune,
                use_task_guided_fusion=use_task_guided_fusion)
    cnn.to(device)

    # 知识蒸馏
    # 教师模型提供“软标签”指导学生学习
    # ========== 添加位置1：创建教师模型 ==========
    teacher = pyCNN(FM=FM, NC=NC, Classes=Classes, para_tune=para_tune,
                    use_task_guided_fusion=use_task_guided_fusion)
    teacher.to(device)
    # 复制学生参数到教师模型（初始化相同）
    for t_param, s_param in zip(teacher.parameters(), cnn.parameters()):
        t_param.data.copy_(s_param.data)

    # 冻结教师模型参数（不更新梯度）
    for param in teacher.parameters():
        param.requires_grad = False
    ema_decay = 0.999  # EMA衰减系数
    # ============================================

    # 计算类别权重（处理类别不平衡问题）
    all_labels = []
    for _, _, labels in train_loader:
        all_labels.extend(labels.cpu().numpy())
    class_counts = np.bincount(all_labels, minlength=Classes)
    class_weights = 1. / (class_counts + 1e-6)
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ========== 添加位置2：定义蒸馏损失函数 ==========
    def distillation_loss(student_logits, teacher_logits, temperature=3.0):
        soft_targets = F.softmax(teacher_logits / temperature, dim=1)
        soft_prob = F.log_softmax(student_logits / temperature, dim=1)
        return F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)

    distill_weight = 0.5
    temperature = 3.0
    # =============================================

    optimizer = optim.Adam(cnn.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    total_loss = 0

    for epoch in range(epochs):
        cnn.train()
        teacher.eval()  # ========== 添加位置3：教师模型设为eval模式 ==========

        for i, (hsi, lidar, target) in enumerate(train_loader):
            hsi, lidar = hsi.to(device), lidar.to(device)
            target = target.to(device)

            # 如果使用任务驱动融合，传入target参数
            if use_task_guided_fusion:
                out1, out2, out3 = cnn(hsi, lidar, target=target)
            else:
                out1, out2, out3 = cnn(hsi, lidar)

            # ========== 添加位置4：教师模型前向传播 ==========
            with torch.no_grad():
                t_out1, t_out2, t_out3 = teacher(hsi, lidar)

            # ========== 添加位置5：计算蒸馏损失 ==========
            loss1_ce = criterion(out1, target)
            loss2_ce = criterion(out2, target)
            loss3_ce = criterion(out3, target)

            loss1_distill = distillation_loss(out1, t_out1, temperature)
            loss2_distill = distillation_loss(out2, t_out2, temperature)
            loss3_distill = distillation_loss(out3, t_out3, temperature)

            # ========== 添加位置6：组合损失 ==========
            loss = loss1_ce + loss2_ce + loss3_ce + distill_weight * (loss1_distill + loss2_distill + loss3_distill)
            # =======================================

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ========== 添加位置7：更新教师模型（EMA） ==========
            # 教师可以看成过去经验的平滑版本
            with torch.no_grad():
                for t_param, s_param in zip(teacher.parameters(), cnn.parameters()):
                    t_param.data = ema_decay * t_param.data + (1 - ema_decay) * s_param.data
            # =================================================

            total_loss += loss.item()

        print('[Epoch: %d] [loss avg: %.4f] [current loss: %.4f]' %
              (epoch + 1, total_loss / (epoch + 1), loss.item()))
        scheduler.step()

    print('Finished Training')
    return cnn


def test(device, net, test_loader):
    """测试模型"""
    net.eval()
    y_pred_test = []
    y_test = []

    for hsi, lidar, labels in test_loader:
        hsi = hsi.to(device)
        lidar = lidar.to(device)
        outputs = net(hsi, lidar)
        outputs = 1 * outputs[2]
        preds = torch.max(outputs, 1)[1].cpu().numpy()

        y_pred_test.extend(preds)
        y_test.extend(labels.numpy())

    return np.array(y_pred_test), np.array(y_test)


def run_single_experiment(dataset_name=DATASET_NAME, train_num=TRAIN_NUM, seed=42, use_task_guided_fusion=True):
    """运行单次实验"""
    print(f"\n{'=' * 70}")
    print(f"开始实验 - 数据集: {dataset_name}, 种子: {seed}, 每类训练样本数: {train_num}")
    print(f"使用任务驱动融合: {use_task_guided_fusion}")
    print(f"{'=' * 70}")

    # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 创建数据加载器
    trainset, testset, TrainLabel1, TestLabel1, NC = create_data_loader(
        dataset_name=dataset_name, train_num=train_num, seed=seed
    )

    # 获取类别数
    Classes = len(torch.unique(TrainLabel1))

    print(f"类别数: {Classes}, 特征通道数: {NC}")
    print(f"训练样本数: {len(trainset)}, 测试样本数: {len(testset)}")

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

    # 训练模型
    tic1 = time.perf_counter()
    net = train(train_loader, EPOCH, Classes, NC, device, use_task_guided_fusion=use_task_guided_fusion)
    toc1 = time.perf_counter()

    # 测试模型
    tic2 = time.perf_counter()
    with torch.no_grad():
        y_pred_test, y_test = test(device, net, test_loader)
    toc2 = time.perf_counter()

    # 计算评估指标
    classification, oa, confusion, each_acc, aa, kappa, used_labels = acc_reports(
        y_test, y_pred_test, dataset=dataset_name
    )

    # 转换格式
    oa = oa / 100.0
    aa = aa / 100.0
    kappa = kappa / 100.0

    # 保存完整each_acc
    each_acc_full = np.zeros(Classes)
    each_acc_full[used_labels] = each_acc / 100.0

    # 计算时间
    TRAINING_TIME = toc1 - tic1
    TESTING_TIME = toc2 - tic2

    # 计算最佳准确率
    BestAcc = oa

    # 保存结果
    save_experiment_results(
        dataset_name=dataset_name,
        train_num=train_num,
        seed=seed,
        y_pred_test=y_pred_test,
        y_test=y_test,
        OA=oa,
        AA=aa,
        Kappa=kappa,
        EachAcc=each_acc_full,
        TRAINING_TIME=TRAINING_TIME,
        TESTING_TIME=TESTING_TIME,
        Train_size=len(trainset),
        Test_size=len(testset),
        Classes=Classes,
        BestAcc=BestAcc,
        classification=classification
    )

    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'seed': seed,
        'BestAcc': BestAcc,
        'OA': oa,
        'AA': aa,
        'Kappa': kappa,
        'EachAcc': each_acc_full,
        'TRAINING_TIME': TRAINING_TIME,
        'TESTING_TIME': TESTING_TIME,
        'Train_size': len(trainset),
        'Test_size': len(testset),
        'Classes': Classes
    }


def save_experiment_results(dataset_name, train_num, seed, y_pred_test, y_test,
                            OA, AA, Kappa, EachAcc, TRAINING_TIME, TESTING_TIME,
                            Train_size, Test_size, Classes, BestAcc, classification):
    """保存实验结果"""
    # 创建实验文件夹
    experiment_folder = create_experiment_folder(dataset_name, train_num)

    # 保存详细结果到mat文件
    detailed_results = {
        'pred_y': y_pred_test,
        'true_y': y_test,
        'OA': OA,
        'AA': AA,
        'Kappa': Kappa,
        'EachAcc': EachAcc,
        'TRAINING_TIME': TRAINING_TIME,
        'TESTING_TIME': TESTING_TIME,
        'BestAcc': BestAcc,
        'seed': seed,
        'train_num': train_num,
        'dataset': dataset_name,
        'Train_size': Train_size,
        'Test_size': Test_size,
        'Classes': Classes,
        'classification_report': str(classification)
    }

    # 保存MAT文件
    mat_filename = os.path.join(experiment_folder, f'seed_{seed}_results.mat')
    savemat(mat_filename, detailed_results)
    print(f"MAT结果文件已保存到: {mat_filename}")

    # 保存为JSON文件
    json_filename = os.path.join(experiment_folder, f'seed_{seed}_results.json')
    json_results = {
        'pred_y': y_pred_test.tolist(),
        'true_y': y_test.tolist(),
        'OA': float(OA),
        'AA': float(AA),
        'Kappa': float(Kappa),
        'EachAcc': EachAcc.tolist(),
        'TRAINING_TIME': float(TRAINING_TIME),
        'TESTING_TIME': float(TESTING_TIME),
        'BestAcc': float(BestAcc),
        'seed': int(seed),
        'train_num': int(train_num),
        'dataset': dataset_name,
        'Train_size': int(Train_size),
        'Test_size': int(Test_size),
        'Classes': int(Classes)
    }

    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=4, ensure_ascii=False)
    print(f"JSON结果文件已保存到: {json_filename}")

    print(f"\n{'=' * 70}")
    print(f"实验结果 (数据集: {dataset_name}, 种子: {seed})")
    print(f"{'-' * 70}")
    print(f'最佳准确率: {BestAcc:.6f}')
    print(f'OA: {OA:.4f}')
    print(f'AA: {AA:.4f}')
    print(f'Kappa: {Kappa:.4f}')
    print(f'训练时间: {TRAINING_TIME:.2f}秒')
    print(f'测试时间: {TESTING_TIME:.2f}秒')
    print(f"{'=' * 70}")


def run_10_experiments(use_task_guided_fusion=True):
    """运行10次实验（种子1-10）"""
    print(f"{'=' * 80}")
    print(f"多模态遥感图像分类实验")
    print(f"数据集: {DATASET_NAME}")
    print(f"每类训练样本数: {TRAIN_NUM}")
    print(f"实验次数: 10 (种子: 1-10)")
    print(f"使用任务驱动融合: {use_task_guided_fusion}")
    print(f"{'=' * 80}")

    # 创建实验文件夹
    experiment_folder = create_experiment_folder(DATASET_NAME, TRAIN_NUM)
    print(f"实验文件夹: {experiment_folder}")

    # 检查索引文件是否存在
    first_index_file = f'./Results/{DATASET_NAME}/train_test_gt_1.mat'
    if not os.path.exists(first_index_file):
        print(f"\n警告: 索引文件不存在!")
        print(f"请先运行 data_prepare.py 生成索引文件:")
        print(f"python data_prepare.py")
        print(f"或者修改 DATASET_NAME 参数")
        return

    # 运行所有实验
    all_results = []
    KAPPA = []
    OA = []
    AA = []
    TRAINING_TIME = []
    TESTING_TIME = []
    ELEMENT_ACC = []

    for exp_id in range(1, 11):  # 种子1-10
        seed = exp_id

        try:
            print(f"\n{'#' * 80}")
            print(f"实验 {exp_id}/10 - 种子: {seed}")
            print(f"{'#' * 80}")

            result = run_single_experiment(
                dataset_name=DATASET_NAME,
                train_num=TRAIN_NUM,
                seed=seed,
                use_task_guided_fusion=use_task_guided_fusion
            )
            all_results.append(result)

            # 收集统计信息
            KAPPA.append(result['Kappa'])
            OA.append(result['OA'])
            AA.append(result['AA'])
            TRAINING_TIME.append(result['TRAINING_TIME'])
            TESTING_TIME.append(result['TESTING_TIME'])
            ELEMENT_ACC.append(result['EachAcc'])

        except Exception as e:
            print(f"\n实验 {exp_id} (种子: {seed}) 失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 汇总所有实验结果
    if all_results:
        # 转换ELEMENT_ACC为合适的形状
        ELEMENT_ACC_array = np.zeros((len(all_results), len(ELEMENT_ACC[0])))
        for i, acc in enumerate(ELEMENT_ACC):
            ELEMENT_ACC_array[i, :] = acc

        # 保存汇总结果
        save_summary_results(
            dataset_name=DATASET_NAME,
            train_num=TRAIN_NUM,
            experiment_folder=experiment_folder,
            all_results=all_results,
            KAPPA=KAPPA,
            OA=OA,
            AA=AA,
            ELEMENT_ACC=ELEMENT_ACC_array,
            TRAINING_TIME=TRAINING_TIME,
            TESTING_TIME=TESTING_TIME
        )
    else:
        print("\n所有实验都失败了，请检查错误信息。")


def save_summary_results(dataset_name, train_num, experiment_folder, all_results,
                         KAPPA, OA, AA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME):
    """保存汇总结果"""
    print(f"\n{'=' * 80}")
    print(f"实验汇总结果 (数据集: {dataset_name}, 每类训练样本数: {train_num})")
    print(f"{'=' * 80}")

    # 打印每次实验的结果
    print(
        f"{'实验ID':<8} {'种子':<8} {'OA':<10} {'AA':<10} {'Kappa':<10} {'训练时间':<12} {'测试时间':<12}")
    print(f"{'-' * 90}")

    for i, result in enumerate(all_results):
        print(f"{i + 1:<8} {result['seed']:<8} "
              f"{result['OA']:.4f}     {result['AA']:.4f}     "
              f"{result['Kappa']:.4f}     {result['TRAINING_TIME']:.2f}     "
              f"{result['TESTING_TIME']:.2f}")

    print(f"{'-' * 90}")

    # 计算统计量
    oa_mean = np.mean(OA)
    oa_std = np.std(OA)
    aa_mean = np.mean(AA)
    aa_std = np.std(AA)
    kappa_mean = np.mean(KAPPA)
    kappa_std = np.std(KAPPA)
    train_time_mean = np.mean(TRAINING_TIME)
    test_time_mean = np.mean(TESTING_TIME)

    # 计算每个类别的平均准确率
    avg_each_acc = np.mean(ELEMENT_ACC, axis=0)
    std_each_acc = np.std(ELEMENT_ACC, axis=0)

    print(f"平均值: "
          f"{oa_mean:.4f} ± {oa_std:.4f}     "
          f"{aa_mean:.4f} ± {aa_std:.4f}     "
          f"{kappa_mean:.4f} ± {kappa_std:.4f}")
    print(f"平均训练时间: {train_time_mean:.2f}秒")
    print(f"平均测试时间: {test_time_mean:.2f}秒")
    print(f"{'=' * 80}")

    # 保存汇总结果到MAT文件
    summary_mat = {
        'dataset': dataset_name,
        'train_num': train_num,
        'num_experiments': len(all_results),
        'seeds': [r['seed'] for r in all_results],
        'OAs': OA,
        'AAs': AA,
        'Kappas': KAPPA,
        'EachAccs': ELEMENT_ACC,
        'Train_times': TRAINING_TIME,
        'Test_times': TESTING_TIME,
        'OA_mean': oa_mean,
        'OA_std': oa_std,
        'AA_mean': aa_mean,
        'AA_std': aa_std,
        'Kappa_mean': kappa_mean,
        'Kappa_std': kappa_std,
        'Train_time_mean': train_time_mean,
        'Test_time_mean': test_time_mean,
        'Avg_EachAcc': avg_each_acc,
        'Std_EachAcc': std_each_acc
    }

    mat_filename = os.path.join(experiment_folder, 'summary_results.mat')
    savemat(mat_filename, summary_mat)
    print(f"\n汇总MAT结果文件已保存到: {mat_filename}")

    # 保存汇总结果到JSON文件
    summary_json = {
        'dataset': dataset_name,
        'train_num': int(train_num),
        'num_experiments': int(len(all_results)),
        'seeds': [int(r['seed']) for r in all_results],
        'OAs': [float(oa) for oa in OA],
        'AAs': [float(aa) for aa in AA],
        'Kappas': [float(kappa) for kappa in KAPPA],
        'EachAccs': ELEMENT_ACC.tolist(),
        'Train_times': [float(t) for t in TRAINING_TIME],
        'Test_times': [float(t) for t in TESTING_TIME],
        'OA_mean': float(oa_mean),
        'OA_std': float(oa_std),
        'AA_mean': float(aa_mean),
        'AA_std': float(aa_std),
        'Kappa_mean': float(kappa_mean),
        'Kappa_std': float(kappa_std),
        'Train_time_mean': float(train_time_mean),
        'Test_time_mean': float(test_time_mean),
        'Avg_EachAcc': avg_each_acc.tolist(),
        'Std_EachAcc': std_each_acc.tolist()
    }

    json_filename = os.path.join(experiment_folder, 'summary_results.json')
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=4, ensure_ascii=False)
    print(f"汇总JSON结果文件已保存到: {json_filename}")

    # ==================== 保存为TXT文件（重要！）====================
    txt_filename = os.path.join(experiment_folder, 'summary_results.txt')
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write(f"{'=' * 80}\n")
        f.write(f"多模态遥感图像分类实验汇总结果\n")
        f.write(f"{'=' * 80}\n\n")

        f.write(f"实验基本信息:\n")
        f.write(f"{'-' * 50}\n")
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"每类训练样本数: {train_num}\n")
        f.write(f"实验次数: {len(all_results)}次\n")
        f.write(f"随机种子: {[r['seed'] for r in all_results]}\n\n")

        f.write(f"详细实验结果:\n")
        f.write(f"{'-' * 50}\n")
        f.write(
            f"{'实验ID':<8} {'种子':<8} {'OA':<10} {'AA':<10} {'Kappa':<10} {'训练时间(s)':<12} {'测试时间(s)':<12}\n")
        f.write(f"{'-' * 90}\n")

        for i, result in enumerate(all_results):
            f.write(f"{i + 1:<8} {result['seed']:<8} "
                    f"{result['OA']:.6f}  {result['AA']:.6f}  "
                    f"{result['Kappa']:.6f}  {result['TRAINING_TIME']:10.2f}  "
                    f"{result['TESTING_TIME']:10.2f}\n")

        f.write(f"{'-' * 90}\n\n")

        f.write(f"统计结果:\n")
        f.write(f"{'-' * 50}\n")
        f.write(f"OA平均值: {oa_mean:.6f} ± {oa_std:.6f}\n")
        f.write(f"AA平均值: {aa_mean:.6f} ± {aa_std:.6f}\n")
        f.write(f"Kappa平均值: {kappa_mean:.6f} ± {kappa_std:.6f}\n")
        f.write(f"平均训练时间: {train_time_mean:.2f} 秒\n")
        f.write(f"平均测试时间: {test_time_mean:.2f} 秒\n\n")

        # 保存每个类别的平均准确率
        f.write(f"每类别平均准确率:\n")
        f.write(f"{'-' * 50}\n")
        for cls_idx in range(len(avg_each_acc)):
            f.write(f"类别 {cls_idx + 1}: {avg_each_acc[cls_idx]:.6f} ± {std_each_acc[cls_idx]:.6f}\n")

        f.write(f"\n{'=' * 80}\n")
        f.write(f"详细数据说明:\n")
        f.write(f"{'-' * 50}\n")
        f.write(f"1. OA (Overall Accuracy): 总体准确率\n")
        f.write(f"2. AA (Average Accuracy): 平均准确率\n")
        f.write(f"3. Kappa: Kappa系数\n")
        f.write(f"4. EachAcc: 每个类别的准确率\n")
        f.write(f"5. 时间单位为秒(s)\n")
        f.write(f"6. 格式: 平均值 ± 标准差\n")
        f.write(f"{'=' * 80}\n")

    print(f"汇总TXT结果文件已保存到: {txt_filename}")

    # ==================== 保存简化版TXT文件（仅关键结果）====================
    simple_txt_filename = os.path.join(experiment_folder, 'key_results.txt')
    with open(simple_txt_filename, 'w', encoding='utf-8') as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"{dataset_name}数据集 (每类{train_num}个训练样本) 关键结果\n")
        f.write(f"{'=' * 60}\n\n")

        f.write(f"OA = {oa_mean:.4f} ± {oa_std:.4f}\n")
        f.write(f"AA = {aa_mean:.4f} ± {aa_std:.4f}\n")
        f.write(f"Kappa = {kappa_mean:.4f} ± {kappa_std:.4f}\n\n")

        f.write(f"详细OA数据:\n")
        for i, oa_value in enumerate(OA):
            f.write(f"  种子 {all_results[i]['seed']}: {oa_value:.4f}\n")

        f.write(f"\n每类准确率:\n")
        for cls_idx in range(len(avg_each_acc)):
            f.write(f"  类别 {cls_idx + 1}: {avg_each_acc[cls_idx]:.4f} ± {std_each_acc[cls_idx]:.4f}\n")

    print(f"关键结果TXT文件已保存到: {simple_txt_filename}")

    # 打印文件结构
    print(f"\n{'=' * 80}")
    print(f"实验结果文件结构:")
    print(f"{'=' * 80}")
    print(f"实验文件夹: {experiment_folder}")
    for seed in range(1, 11):
        mat_file = os.path.join(experiment_folder, f'seed_{seed}_results.mat')
        if os.path.exists(mat_file):
            print(f"├── seed_{seed}_results.mat")
            print(f"├── seed_{seed}_results.json")
    print(f"├── summary_results.mat")
    print(f"├── summary_results.json")
    print(f"├── summary_results.txt (详细汇总)")
    print(f"└── key_results.txt (关键结果)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    # 配置参数
    USE_TASK_GUIDED_FUSION = True  # 是否使用任务驱动的特征融合修正

    # 运行10次实验
    run_10_experiments(use_task_guided_fusion=USE_TASK_GUIDED_FUSION)