import pandas as pd
import numpy as np

# 加载原始CSV文件
data = pd.read_csv('train_large.csv')

# 获取数据集的总行数
total_rows = len(data)

# 定义划分比例
train_ratio = 0.6
test_ratio = 0.2
validation_ratio = 0.2

# 计算划分后的数据集大小
train_size = int(total_rows * train_ratio)
test_size = int(total_rows * test_ratio)
validation_size = int(total_rows * validation_ratio)

# 创建随机索引，并确保不重复
indices = np.random.permutation(total_rows)

# 划分数据集
train_indices = indices[:train_size]
test_indices = indices[train_size:(train_size + test_size)]
validation_indices = indices[(train_size + test_size):]

# 根据索引划分数据集
train_set = data.iloc[train_indices]
test_set = data.iloc[test_indices]
validation_set = data.iloc[validation_indices]

# 从原始数据中删除已经划分的数据
data.drop(indices[:train_size], inplace=True)
data.drop(indices[train_size:(train_size + test_size)], inplace=True)
data.drop(indices[(train_size + test_size):], inplace=True)

# 保存划分后的数据集为CSV文件
train_set.to_csv('train_set.csv', index=False)
test_set.to_csv('test_set.csv', index=False)
validation_set.to_csv('validation_set.csv', index=False)

# 保存剩余的数据到新的CSV文件
data.to_csv('remaining_data.csv', index=False)
