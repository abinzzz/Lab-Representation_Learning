import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义卷积自动编码器类
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 7)
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建模型实例并移至设备
conv_autoencoder = ConvAutoencoder().to(device)

# 数据加载器
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(conv_autoencoder.parameters(), lr=1e-3)

# 训练模型
num_epochs = 10
train_losses = []

for epoch in range(num_epochs):
    conv_autoencoder.train()  # 设置为训练模式
    total_train_loss = 0
    for data in train_loader:
        img, _ = data
        img = img.to(device)
        # 前向传播
        output = conv_autoencoder(img)
        loss = criterion(output, img)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, num_epochs, avg_train_loss))

# 测试模型
conv_autoencoder.eval()  # 设置为评估模式
total_test_loss = 0
test_losses = []

with torch.no_grad():
    for data in test_loader:
        img, _ = data
        img = img.to(device)
        output = conv_autoencoder(img)
        loss = criterion(output, img)
        total_test_loss += loss.item()
avg_test_loss = total_test_loss / len(test_loader)
test_losses.append(avg_test_loss)

print('Test Loss: {:.4f}'.format(avg_test_loss))

# 绘制训练loss图像
plt.plot(train_losses, label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 可视化函数
def visualize_output(autoencoder, data_loader, num_images=10):
    autoencoder.eval()  # 设置为评估模式
    images, _ = next(iter(data_loader))
    images = images.to(device)
    output = autoencoder(images)
    images = images.cpu().numpy()
    output = output.detach().cpu().numpy()

    fig, axes = plt.subplots(nrows=2, ncols=num_images, figsize=(10, 2))
    for i in range(num_images):
        # 显示原始图像
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        # 显示重建图像
        axes[1, i].imshow(output[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
    plt.show()

# 在训练结束后调用可视化函数
visualize_output(conv_autoencoder, train_loader)


