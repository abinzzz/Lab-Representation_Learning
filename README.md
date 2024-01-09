


## 项目简介
此项目涉及使用机器学习技术处理和分析MNIST手写数字数据集。包含两种类型的自动编码器实现（卷积神经网络和多层感知机）和一个主成分分析（PCA）脚本。

## 目录结构
```
.
├── autoencoder_cnn.py      # 卷积神经网络自动编码器
├── autoencoder_mlp.py      # 多层感知机自动编码器
├── data
│   └── MNIST               # MNIST数据集存放目录
├── mnist-pca.py            # 执行MNIST数据集的PCA分析
└── mnist_train.csv         # MNIST训练数据集
```

## 文件说明
- `autoencoder_cnn.py`：使用卷积神经网络（CNN）构建的自动编码器，适用于图像数据的特征提取和压缩。
- `autoencoder_mlp.py`：使用多层感知机（MLP）构建的自动编码器，同样用于特征提取和数据压缩。
- `mnist-pca.py`：利用主成分分析（PCA）对MNIST数据集进行降维处理，用于数据探索和可视化。
- `mnist_train.csv`：MNIST训练数据集，包含手写数字图像的标签和像素值。

## 需要的环境
- Python
- PyTorch
- Numpy
- Pandas
- Matplotlib

## 运行指南
1. 确保安装了Python及相关库（如PyTorch、pandas、numpy等）。
2. 将MNIST数据集放置在`data/MNIST`目录下。
3. 运行自动编码器脚本（卷积神经网络或多层感知机）来训练模型并提取特征。
4. 运行`mnist-pca.py`进行PCA分析和数据可视化。

## 注意事项
- 确保在运行脚本前已正确安装所有依赖。
- 对于大型数据集，特别是在使用卷积自动编码器时，建议使用具备足够计算能力的硬件。

