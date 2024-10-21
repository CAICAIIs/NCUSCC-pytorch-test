import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler  # 正确的导入方式
import time
import psutil
import os

# 数据集放置路径
data_save_pth = "./data"

# 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 使用官方方式加载数据集
trainset = torchvision.datasets.CIFAR10(root=data_save_pth, train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=data_save_pth, train=False, download=True, transform=transform)

# 定义CNN模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型、优化器和损失函数
model = Net().cuda()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss().cuda()
scaler = GradScaler()  # 初始化GradScaler

# 训练和评估模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 测试不同的batch size
batch_sizes = [32, 64, 128, 256]  # 不同的batch size
for batch_size in batch_sizes:
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    model.train()
    start_time = time.time()
    memory_usage_start = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    
    for epoch in range(10):  # 使用10个epoch作为示例
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast(device_type='cuda'):  # 正确的autocast使用方式
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Batch size: {batch_size}')
    print(f'Training Time (seconds): {elapsed_time:.2f}')
    print(f'Initial Memory Usage (MB): {memory_usage_start:.2f}')
    print(f'Final Memory Usage (MB): {psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024):.2f}')
    
    # 测试模型性能
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total if total > 0 else 0
    print(f'Accuracy of the network on the test images: {accuracy:.2f}%\n')