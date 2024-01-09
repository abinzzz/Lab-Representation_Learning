import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载器
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)#batchsize

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 可视化函数
def visualize_output(autoencoder, data_loader, num_images=10):
    autoencoder.eval()  # 设置为评估模式
    images, _ = next(iter(data_loader))  # 修改此处
    # 将图像扁平化并转移到设备上
    images_flattened = images.view(images.size(0), -1).to(device)
    # 自动编码器重建图像
    outputs = autoencoder(images_flattened)
    outputs = outputs.view(images.size(0), 1, 28, 28).cpu().data

    # 绘制图像
    fig, axes = plt.subplots(nrows=2, ncols=num_images, figsize=(10, 2))
    for i in range(num_images):
        # 显示原始图像
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        # 显示重建图像
        axes[1, i].imshow(outputs[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
    plt.show()

#定义自动编码器
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28)

        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(28 * 28, 256),
#             nn.ReLU(),
#
#             #nn.Dropout(0.058),
#
#             nn.Linear(256, 32),
#             nn.ReLU(),
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(32, 256),
#             nn.ReLU(),
#
#             nn.Linear(256, 28 * 28)
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

# 定义不同的损失函数
loss_functions = {
    "MSE": nn.MSELoss(),
    "L1": nn.L1Loss(),
    "BCEWithLogits": nn.BCEWithLogitsLoss(),
    "SmoothL1": nn.SmoothL1Loss()
}

autoencoder = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)#学习率,正则化项

# 定义训练函数
def train(model, data_loader, optimizer, criterion, num_epochs=10):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for data in data_loader:
            img, _ = data
            img = img.view(img.size(0), -1).to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}')
    return train_losses

# 定义测试函数
def test(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in data_loader:
            img, _ = data
            img = img.view(img.size(0), -1).to(device)
            output = model(img)
            loss = criterion(output, img)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    return avg_loss

# 训练模型
train_losses = train(autoencoder, train_loader, optimizer, criterion)

# 测试模型
test_loss = test(autoencoder, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}')

# 绘制训练损失变化图
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 在训练后调用可视化函数
visualize_output(autoencoder, train_loader)
