# 导入包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import warnings # 当前版本的Seaborn会产生许多警告，这些警告将被忽略。
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# 将数据集整合到Python笔记本中
d0 = pd.read_csv('mnist_train.csv')
#print(d0.head(5)) # 检查数据


# 从数据集中分离标签
l = d0['label']
d = d0.drop('label', axis=1)


# print(d.head(5))
# print(l.head(5))

# 用形状确认
# print(l.shape)
# print(d.shape)

# 视觉上绘制一个样本数字
# plt.figure(figsize=(5,5))
# idx = 150
#
# grid_data = d.iloc[idx].values.reshape(28,28) # 从1d重塑为2d
# plt.imshow(grid_data, interpolation='none', cmap='gray')
# plt.show()
#
# print('上面的值是', l[idx])


# 创建带有标签的15k数据集
label = l.head(15000)
data = d.head(15000)
#print('数据的形状是', data.shape)

# 数据预处理：标准化数据集
from sklearn.preprocessing import StandardScaler
#standard_data = StandardScaler().fit_transform(data)
standard_data = data
#print(standard_data.shape)

# 创建协方差矩阵的样本数据：A^T * A
sample_data = standard_data

# 使用numpy进行矩阵乘法
covar_matrix = np.matmul(sample_data.T, sample_data)
#print('协方差矩阵的形状 = ', covar_matrix.shape)


# 处理特征向量和特征值
from scipy.linalg import eigh # 来自线性代数的scipy
values, vectors = eigh(covar_matrix, eigvals=(782,783)) # 返回协方差矩阵的值和向量，前两个(782,783)
#print('特征向量的形状是', vectors.shape)
vectors = vectors.T
#print('特征向量更新后的形状是', vectors.shape)


# 通过上述特征向量将784维数据集降维为2维
new_coordinates = np.matmul(vectors, sample_data.T)
#print('结果新数据点的形状是', vectors.shape, 'X', sample_data.T.shape, '=', new_coordinates.shape)


# 将标签附加到2维投影的新数据集上
new_coordinates = np.vstack((new_coordinates, label)).T
#print('新数据集的形状是', new_coordinates.shape)


# 创建数据框
matrix_df = pd.DataFrame(data=new_coordinates, columns=('1st_principal', '2nd_principal', 'labels'))
#print(matrix_df.head(5))


# sn.FacetGrid(matrix_df, hue='labels', height=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
# plt.show()


# 使用SKlearn导入PCA
from sklearn import decomposition
pca = decomposition.PCA()


# 用PCA进行降维（非可视化）

pca.n_components = 784
pca_data = pca.fit_transform(sample_data)

percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);

cum_var_explained = np.cumsum(percentage_var_explained)

# 绘制PCA谱
plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained, linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()


# 如果我们取200维，大约90%的方差可以解释。

# 直接输入参数
pca.n_components = 2
pca_data = pca.fit_transform(sample_data)

print('pca降维数据的形状 = ', pca_data.shape)

# 数据调整 - 向缩减的矩阵中添加标签列
pca_data = np.vstack((pca_data.T, label)).T

# 数据框和绘制PCA数据
pca_df = pd.DataFrame(data=pca_data, columns=('1st_principal', '2nd_principal', 'labels'))
sn.FacetGrid(pca_df, hue='labels', height=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()



# # 导入必要的包
# from sklearn.decomposition import KernelPCA
#
# # 选择 Kernel PCA 参数
# kernel_pca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
#
# standard_data = StandardScaler().fit_transform(standard_data)
#
# # 对标准化后的数据应用 Kernel PCA
# kernel_pca_data = kernel_pca.fit_transform(standard_data)
#
# # 将标签附加到 Kernel PCA 降维后的数据上
# kernel_pca_data = np.vstack((kernel_pca_data.T, label)).T
#
# # 创建包含 Kernel PCA 数据的 DataFrame
# kernel_pca_df = pd.DataFrame(data=kernel_pca_data, columns=('1st_principal', '2nd_principal', 'labels'))
#
# # 可视化 Kernel PCA 降维结果
# sn.FacetGrid(kernel_pca_df, hue='labels', height=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
# plt.title('MNIST Data Reduced to 2D with Kernel PCA')
# plt.show()
#
