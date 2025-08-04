import random
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io as sio
from numpy import resize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time
import record


def loadData():

    # data = sio.loadmat(
    #     r'./Data/Muufl_hsi.mat')[
    #     'hsi']
    # labels = sio.loadmat(
    #     r'.\Data\Muufl_gt.mat')[
    #     'Muufl_gt']
    # lidar_data = sio.loadmat(
    #     r'.\Data\Muufl_Lidar.mat')[
    #     'lidar']

    data = sio.loadmat(
        r'./Data/Houston.mat')[
        'img']
    labels = sio.loadmat(
        r'./Data/Houston_gt.mat')[
        'Houston_gt']
    lidar_data = sio.loadmat(
        r'./Data/Houston_LiDAR.mat')[
        'img']

    # data = sio.loadmat(
    #     r'./Data/Trento_hsi.mat')[
    #     'HSI']
    # labels = sio.loadmat(
    #     r'./Data/Trento_allgrd.mat')[
    #     'mask_test']
    # lidar_data = sio.loadmat(
    #     r'./Data/Trento_LiDAR.mat')[
    #     'LiDAR']

    # data = sio.loadmat(
    #     r'C:\Users\liuliu\Desktop\应用中心\论文\论文（已看）\CSCA\CSCANet_main\Data\augsburg_hsi.mat')[
    #     'augsburg_hsi']
    # labels = sio.loadmat(
    #     r'C:\Users\liuliu\Desktop\应用中心\论文\论文（已看）\CSCA\CSCANet_main\Data\augsburg_gt.mat')[
    #     'augsburg_gt']
    # lidar_data = sio.loadmat(
    #     r'C:\Users\liuliu\Desktop\应用中心\论文\论文（已看）\CSCA\CSCANet_main\Data\augsburg_sar.mat')[
    #     'augsburg_sar']

    # 检查LiDAR数据的维度
    if len(lidar_data.shape) == 3 and lidar_data.shape[2] >= 2:
        print(f"原始LiDAR数据形状: {lidar_data.shape} - 将对两个通道取平均值")
        # 对两个通道取平均值，降维为单通道
        lidar_data = np.mean(lidar_data, axis=2)
    elif len(lidar_data.shape) == 2:
        print(f"原始LiDAR数据形状: {lidar_data.shape} - 已经是单通道")
    else:
        raise ValueError(f"意外的LiDAR数据形状: {lidar_data.shape}")

    # 打印处理后的形状
    print(f"处理后LiDAR数据形状: {lidar_data.shape}")

    return data, labels, lidar_data


# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(hsi_data, lidar_data, y, windowSize=5, removeZeroLabels=True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    hsi_padded = padWithZeros(hsi_data, margin=margin)
    lidar_padded = padWithZeros(np.expand_dims(lidar_data, axis=-1), margin=margin)
    # split patches
    patchesData_hsi = np.zeros((hsi_data.shape[0] * hsi_data.shape[1], windowSize, windowSize, hsi_data.shape[2]))
    patchesData_lidar = np.zeros((lidar_data.shape[0] * lidar_data.shape[1], windowSize, windowSize, 1))
    patchesLabels = np.zeros((hsi_data.shape[0] * hsi_data.shape[1]))

    patchIndex = 0
    for r in range(margin, hsi_padded.shape[0] - margin):
        for c in range(margin, hsi_padded.shape[1] - margin):
            patchesData_hsi[patchIndex] = hsi_padded[r - margin:r + margin + 1, c - margin:c + margin + 1, :]
            patchesData_lidar[patchIndex] = lidar_padded[r - margin:r + margin + 1, c - margin:c + margin + 1, :]
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex += 1
    if removeZeroLabels:
        mask = patchesLabels > 0
        return patchesData_hsi[mask], patchesData_lidar[mask], patchesLabels[mask] - 1

    return patchesData_hsi, patchesData_lidar, patchesLabels

# 封装为可迭代的Dataset对象
""" Training dataset"""
class TrainDS(torch.utils.data.Dataset):
    def __init__(self, Xtrain_hsi, Xtrain_lidar, ytrain):
        self.len = ytrain.shape[0]
        self.x_hsi = torch.FloatTensor(Xtrain_hsi).squeeze(1)  # HSI保持[N,30,13,13]
        self.x_lidar = torch.FloatTensor(Xtrain_lidar)  # 不要squeeze，保持[N,1,13,13]
        self.y = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        return self.x_hsi[index], self.x_lidar[index], self.y[index]

    def __len__(self):
        return self.len


""" Testing dataset"""
class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest_hsi, Xtest_lidar, ytest):
        self.len = ytest.shape[0]
        self.x_hsi = torch.FloatTensor(Xtest_hsi)
        self.x_lidar = torch.FloatTensor(Xtest_lidar)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_hsi[index], self.x_lidar[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len



def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def acc_reports(y_test, y_pred_test):
    # Houston2013
    all_names = ['Healthy Grass', 'Stressed Grass', 'Synthetic Grass', 'Tree',
                 'Soil', 'Water', 'Residential', 'Commercial', 'Road', 'Highway',
                 'Railway', 'Parking Lot1', 'Parking Lot2', 'Tennis Court',
                 'Running Track']

    # Trento
    # all_names = ['Apple Trees', 'Buildings', 'Ground',
    #              'Woods', 'Vineyard', 'Roads', ]  # Trento 6

    # Muufl
    # all_names = ['Trees', 'Mostly grass', 'Mixed ground surface',
    #              'Dirt and sand', 'Road','Water', 'Building Shadow',
    #              'Building','Sidewalk', 'Yellow curb','Cloth panels']  # Muufl 11

    # Augsburg
    # all_names = ['Forest', 'Residential Area', 'Industrial Area',
    #     'Low Plants', 'Soil', 'Allotment',
    #     'Commercial Area', 'Water', 'Railway',
    #     'Harbor', 'Pasture', 'Roads','Urban Green'] # Augsburg 13

    unique_labels = sorted(np.unique(y_test))
    used_names = [all_names[i] for i in unique_labels]

    classification = classification_report(
        y_test,
        y_pred_test,
        labels=unique_labels,
        digits=4,
        target_names=used_names
    )

    confusion = confusion_matrix(y_test, y_pred_test, labels=unique_labels)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    oa = accuracy_score(y_test, y_pred_test)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100, unique_labels

