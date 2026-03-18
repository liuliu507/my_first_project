import numpy as np
import scipy.io as sio


def samplingFixedNum(sample_num, groundTruth, seed):
    """
    实验室提供的采样方法
    """
    labels_loc = {}
    train_ = {}
    test_ = {}

    np.random.seed(seed)
    m = max(groundTruth)

    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        train_[i] = indices[:sample_num]
        test_[i] = indices[sample_num:]

    train_fix_indices = []
    test_fix_indices = []

    for i in range(m):
        train_fix_indices += train_[i]
        test_fix_indices += test_[i]

    np.random.shuffle(train_fix_indices)
    np.random.shuffle(test_fix_indices)

    return train_fix_indices, test_fix_indices


def data_load_and_save(name="Muufl", train_num=20):
    """
    加载数据并保存采样结果
    参数说明:
    参数说明:
        name: 数据集名称
        train_num: 每类训练样本数
    """

    # 根据数据集名称加载标签数据
    if name == "Trento":
        # 加载标签数据
        AllLabel = sio.loadmat(r'./Data/Trento_allgrd.mat')['mask_test']

    elif name == "Muufl":
        AllLabel = sio.loadmat(r'./Data/Muufl_gt.mat')['Muufl_gt']

    elif name == "Houston":
        AllLabel = sio.loadmat(r'./Data/Houston_gt.mat')['Houston_gt']

    elif name == "Augsburg":
        AllLabel = sio.loadmat(r'./Data/augsburg_gt.mat')['augsburg_gt']
    else:
        raise ValueError(f"不支持的数据库: {name}")

    print(f"数据集: {name}")
    print(f"标签数据形状: {AllLabel.shape}")
    print(f"每类训练样本数: {train_num}")

    # 进行10次实验
    for i in range(10):
        seed = i + 1  # 随机种子1-10，与第二个代码完全一致

        # 展平标签数据
        gt = AllLabel.reshape(np.prod(AllLabel.shape[:2]), ).astype(np.int64)

        # 使用与第二个代码完全相同的采样函数
        train_index, test_index = samplingFixedNum(train_num, gt, seed)

        # 创建训练和测试数据数组 - 与第二个代码逻辑完全一致
        train_data = np.zeros(np.prod(AllLabel.shape[:2]), )
        train_data[train_index] = gt[train_index]
        test_data = np.zeros(np.prod(AllLabel.shape[:2]), )
        test_data[test_index] = gt[test_index]

        # 重新塑形为原始数据的形状 - 与第二个代码逻辑完全一致
        # 注意：这里使用np.prod确保维度正确
        original_shape = AllLabel.shape
        train_data = train_data.reshape(original_shape[0], original_shape[1])
        test_data = test_data.reshape(original_shape[0], original_shape[1])

        # 保存训练和测试数据到MAT文件 - 格式与第二个代码完全一致
        save_path = f'./Results/{name}/train_test_gt_{i + 1}.mat'

        # 确保保存目录存在
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        sio.savemat(save_path, {
            'train_data': train_data,
            'test_data': test_data,
            'train_index': train_index,
            'test_index': test_index
        })

        print(f"  实验 {i + 1}: 已保存到 {save_path}")
        print(f"    训练样本数: {len(train_index)}")
        print(f"    测试样本数: {len(test_index)}")

        # 输出每类样本分布
        train_labels = train_data.ravel()
        test_labels = test_data.ravel()

        unique_classes = np.unique(gt[gt > 0])
        print(f"    类别分布:")
        for cls in unique_classes:
            train_cls_count = np.sum(train_labels == cls)
            test_cls_count = np.sum(test_labels == cls)
            print(f"      类别 {int(cls)}: 训练={train_cls_count}, 测试={test_cls_count}")


# 使用示例
if __name__ == "__main__":
    # 您可以在这里指定要处理的数据集
    datasets = ["Houston", "Muufl", "Trento", "Augsburg"]

    for dataset_name in datasets:
        try:
            data_load_and_save(name=dataset_name, train_num=20)
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {e}")