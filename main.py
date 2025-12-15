import os
from operator import truediv

import numpy as np
import random
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dataf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from pymodel_base import pyCNN
from data_prepare import loadData, applyPCA, padWithZeros, createImageCubes, TrainDS, TestDS,acc_reports
import record

# 设置参数
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

BATCH_SIZE_TRAIN = 64
EPOCH = 200
LR = 0.001
# CLASSES_NUM = 15  # Houston数据集
CLASSES_NUM = 11  # Muufl数据集
# CLASSES_NUM = 6  # Trento数据集


def create_data_loader():
    """创建数据加载器"""
    # 参数设置
    samples_per_class = 20
    # patch_size = 13
    patch_size = 9
    # patch_size=15
    pca_components = 30

    # 加载数据
    hsi_data, labels, lidar_data = loadData()
    print('Hyperspectral data shape: ', hsi_data.shape)
    print('LiDAR data shape: ', lidar_data.shape)
    print('Label shape: ', labels.shape)

    # PCA降维
    print('\n... PCA transformation ...')
    hsi_pca = applyPCA(hsi_data, numComponents=pca_components)
    print('HSI data shape after PCA: ', hsi_pca.shape)

    # 创建数据立方体
    print('\n... create data cubes ...')
    X_hsi, X_lidar, y_all = createImageCubes(hsi_pca, lidar_data, labels, windowSize=patch_size)
    print('HSI cube shape: ', X_hsi.shape)
    print('LiDAR cube shape: ', X_lidar.shape)
    print('Label shape: ', y_all.shape)

    # PyTorch格式转换
    X_hsi = X_hsi.reshape(-1, patch_size, patch_size, pca_components).transpose(0, 3, 1, 2)  # [N, C, H, W]
    X_lidar = X_lidar.reshape(-1, patch_size, patch_size, 1).transpose(0, 3, 1, 2)  # [N, 1, H, W]
    # 修改LiDAR数据的reshape方式
    X_lidar = X_lidar.reshape(-1, 1, patch_size, patch_size)  # [N,1,13,13]

    # 划分训练集和测试集
    np.random.seed(0)
    classes = np.unique(y_all)
    train_indices = []
    test_indices = []

    for cls in classes:
        cls_indices = np.where(y_all == cls)[0]
        np.random.shuffle(cls_indices)

        min_samples = min(len(cls_indices), samples_per_class)
        train_samples = max(1, min_samples // 2)

        if len(cls_indices) < samples_per_class:
            train_idx = cls_indices[:train_samples]
            test_idx = cls_indices[train_samples:]
        else:
            train_idx = cls_indices[:samples_per_class]
            test_idx = cls_indices[samples_per_class:]

        train_indices.extend(train_idx)
        test_indices.extend(test_idx)

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    Xtrain_hsi = X_hsi[train_indices]
    Xtrain_lidar = X_lidar[train_indices]
    ytrain = y_all[train_indices]

    Xtest_hsi = X_hsi[test_indices]
    Xtest_lidar = X_lidar[test_indices]
    ytest = y_all[test_indices]

    print('Train HSI shape: ', Xtrain_hsi.shape)
    print('Test HSI shape: ', Xtest_hsi.shape)
    print('Train LiDAR shape: ', Xtrain_lidar.shape)
    print('Test LiDAR shape: ', Xtest_lidar.shape)

    # 创建数据集
    trainset = TrainDS(Xtrain_hsi, Xtrain_lidar, ytrain)
    testset = TestDS(Xtest_hsi, Xtest_lidar, ytest)

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

    return train_loader, test_loader, y_all


def train(train_loader, epochs):
    """训练模型"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    para_tune = True
    FM = 64

    cnn = pyCNN(FM=FM, NC=30, Classes=CLASSES_NUM, para_tune=para_tune)
    cnn.to(device)

    # 计算类别权重，处理类别不平衡问题
    all_labels = []

    for _, _, labels in train_loader:
        all_labels.extend(labels.cpu().numpy())
    class_counts = np.bincount(all_labels, minlength=CLASSES_NUM)
    class_weights = 1. / (class_counts + 1e-6)
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(cnn.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)

    total_loss = 0
    BestAcc = 0

    for epoch in range(epochs):
        cnn.train()
        for i, (hsi, lidar, target) in enumerate(train_loader):
            hsi, lidar = hsi.to(device), lidar.to(device)
            target = target.to(device)

            out1, out2, out3 = cnn(hsi, lidar)
            loss1 = criterion(out1, target)
            loss2 = criterion(out2, target)
            loss3 = criterion(out3, target)
            loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print('[Epoch: %d] [loss avg: %.4f] [current loss: %.4f]' %
              (epoch + 1, total_loss / (epoch + 1), loss.item()))
        scheduler.step()
    print('Finished Training')
    return cnn, device


def test(device, net, test_loader):
    """测试模型"""
    net.eval()
    y_pred_test = []
    y_test = []

    for hsi, lidar, labels in test_loader:
        hsi = hsi.to(device)
        lidar = lidar.to(device)
        outputs = net(hsi, lidar)
        outputs = 1 * outputs[2] + 0.01 * outputs[1] + 0.01 * outputs[0]
        preds = torch.max(outputs, 1)[1].cpu().numpy()

        y_pred_test.extend(preds)
        y_test.extend(labels.numpy())

    return np.array(y_pred_test), np.array(y_test)


if __name__ == '__main__':
    # 创建数据加载器
    train_loader, test_loader, y_all = create_data_loader()

    # 初始化记录变量
    ITER = 10
    KAPPA = []
    OA = []
    AA = []
    TRAINING_TIME = []
    TESTING_TIME = []
    ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))

    for index_iter in range(ITER):

        print("iter:", index_iter)
        # 训练模型
        tic1 = time.perf_counter()
        net, device = train(train_loader, epochs=EPOCH)
        toc1 = time.perf_counter()

        # 测试模型
        tic2 = time.perf_counter()
        with torch.no_grad():  # 禁用梯度计算
            y_pred_test, y_test = test(device, net, test_loader)
        toc2 = time.perf_counter()

        # 计算评估指标
        classification, oa, confusion, each_acc, aa, kappa, used_labels = acc_reports(y_test, y_pred_test)

        # 保存分类报告（字符串形式）
        classification_str = str(classification)

        # 记录结果
        KAPPA.append(kappa)
        OA.append(oa)
        AA.append(aa)
        TRAINING_TIME.append(toc1 - tic1)
        TESTING_TIME.append(toc2 - tic2)

        # 保存完整each_acc
        each_acc_full = np.zeros(CLASSES_NUM)
        each_acc_full[used_labels] = each_acc
        ELEMENT_ACC[index_iter, :] = each_acc_full

        record_dir = r"C:\Users\liuliu\Desktop\应用中心\论文\论文（已看）\MS2CA\MS2CANet-main\record"
        output_file = os.path.join(record_dir, "MS2CANet_HC103 Muufl_CPT.txt")


        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    record.record_output(
        OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
        output_file)