import krippendorff
import numpy as np

# 准备数据
# 数据应该是一个二维数组，行代表不同的单元，列代表不同的评估者
# 使用 np.nan 来表示缺失的数据
data2 = np.array([
    [1, 1, np.nan],
    [2, 2, 2],
    [3, 3, np.nan],
    [3, 3, 3],
    [np.nan, np.nan, np.nan],
    [1, 1, 1]
])
value_counts = np.array([[0, 0, 0, 0],
                             [0, 0, 0, 0],
                             [0, 2, 0, 0],
                             [2, 0, 0, 1],
                             [0, 0, 2, 0],
                             [0, 0, 2, 1],
                             [0, 0, 0, 0],
                             [1, 0, 1, 0],
                             [0, 2, 0, 2],
                             [2, 0, 0, 0],
                             [2, 0, 0, 0],
                             [0, 0, 2, 0],
                             [0, 0, 2, 2],
                             [0, 0, 0, 0],
                             [0, 0, 1, 1]])
# 假设我们有3个编码员对10个样本进行了分类，分类标签为A、B、C
data = [
    ['A', 'B', 'A'],  # 样本1的编码结果
    ['B', 'B', 'A'],  # 样本2
    ['A', 'A', 'A'],  
    ['C', 'C', 'C'],
    ['B', 'A', 'B'],
    ['A', 'B', 'B'],
    ['C', 'A', 'C'],
    ['B', 'B', 'B'],
    ['A', 'A', 'B'],
    ['C', 'C', 'A']
]
data1 = np.array([
    [np.nan,0,1],
    [1,0,1],
    [1,np.nan,np.nan],
    [np.nan,0,1],
])
# 计算 Krippendorff's alpha 使用名义数据的度量
alpha_nominal = krippendorff.alpha(reliability_data=data2, level_of_measurement='nominal')

print("Krippendorff's alpha (nominal):", alpha_nominal)

# 如果你的数据是序数数据，可以这样计算：
# alpha_ordinal = krippendorff.alpha(data, level_of_measurement='ordinal')

# 对于间隔和比率数据，可以这样计算：
# alpha_interval = krippendorff.alpha(data, level_of_measurement='interval')
# alpha_ratio = krippendorff.alpha(data, level_of_measurement='ratio')
