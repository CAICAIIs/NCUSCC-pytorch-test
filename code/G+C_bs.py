import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import time

# 定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 数据集路径
data_dir = './data'

# 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载CIFAR-10数据集
train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                             download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                            download=True, transform=transform)

# 创建数据加载器
def create_loader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

# 初始化模型、优化器和损失函数
def init_model(device, batch_size):
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    train_loader = create_loader(train_dataset, batch_size)
    test_loader = create_loader(test_dataset, batch_size, shuffle=False)
    return model, optimizer, criterion, scaler, train_loader, test_loader

# 训练模型
def train_model(model, device, train_loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # 使用自动混合精度，指定device_type
        if device.type == 'cuda':
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 测试模型性能
def test_model(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 主函数
def main():
    batch_sizes = [32, 64, 128]  # 测试不同的批量大小
    for batch_size in batch_sizes:
        print(f"Testing with batch size: {batch_size}")
        
        # 在CPU上测试
        device = torch.device("cpu")
        model_cpu, optimizer_cpu, criterion_cpu, scaler_cpu, train_loader_cpu, test_loader_cpu = init_model(device, batch_size)
        start_time_cpu = time.time()
        train_loss_cpu = train_model(model_cpu, device, train_loader_cpu, optimizer_cpu, criterion_cpu, scaler_cpu)
        test_acc_cpu = test_model(model_cpu, device, test_loader_cpu)
        print(f"CPU - Loss: {train_loss_cpu:.4f}, Accuracy: {test_acc_cpu:.4f}, Time: {time.time() - start_time_cpu:.4f} seconds")
        
        # 在GPU上测试
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model_gpu, optimizer_gpu, criterion_gpu, scaler_gpu, train_loader_gpu, test_loader_gpu = init_model(device, batch_size)
            start_time_gpu = time.time()
            train_loss_gpu = train_model(model_gpu, device, train_loader_gpu, optimizer_gpu, criterion_gpu, scaler_gpu)
            test_acc_gpu = test_model(model_gpu, device, test_loader_gpu)
            print(f"GPU - Loss: {train_loss_gpu:.4f}, Accuracy: {test_acc_gpu:.4f}, Time: {time.time() - start_time_gpu:.4f} seconds")

if __name__ == "__main__":
    main()