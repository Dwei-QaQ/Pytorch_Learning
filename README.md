# PyTorch_Learning

官方教程: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)

PyTorch 文档: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

## 1.PyTorch 基础学习内容

PyTorch 是一个开源的 Python 机器学习库，基于 Torch，用于自然语言处理等应用程序。以下是 PyTorch 的基础学习内容：

### 1.1 PyTorch 基础概念

   -张量 (Tensors): PyTorch 的基本数据结构，类似于 NumPy 的 ndarray，但可以在 GPU 上运行,张量可以看作是多维数组，其维度（dim）称为轴（axis）或阶（rank）

   -自动微分 (Autograd): PyTorch 的自动微分引擎，用于神经网络训练

   -计算图 (Computational Graph): PyTorch 使用动态计算图，也称为 define-by-run 框架

### 1.2 张量操作

```python
import torch

# 创建张量
x = torch.empty(5, 3)                      # 未初始化的张量
x = torch.rand(5, 3)                       # 随机初始化的张量
x = torch.zeros(5, 3, dtype=torch.long)    # 全零张量
x = torch.tensor([5.5, 3])                 # 从数据创建张量

# 张量加法
# 示例1：相同形状
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print(x + y)  # 可以正常相加

# 示例2：可广播的形状
x = torch.tensor([1.0, 2.0, 3.0])  # 形状 (3,)
y = torch.rand(5, 3)               # 形状 (5, 3)
print(x + y)  # x会被广播为(5,3)

print(x.shape)  # 添加这行查看x的形状
print(y.shape)  # 添加这行查看y的形状
print(x + y)

# 改变形状
x = torch.randn(4, 4)
y = x.view(16)  # 展平
z = x.view(-1, 8)  # -1 表示自动计算该维度大小
```

### 1.3  自动微分 (Autograd)

```python
# 创建一个张量并设置 requires_grad=True 来跟踪计算
x = torch.ones(2, 2, requires_grad=True)

# 执行一个张量操作
y = x + 2
z = y * y * 3
out = z.mean()

# 反向传播
out.backward()

# 打印梯度 d(out)/dx
print(x.grad)
```

### 1.4 神经网络 (Neural Networks)基础

PyTorch 提供了 torch.nn 模块来构建神经网络

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 个输入图像通道，6 个输出通道，3x3 卷积核
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 全连接层
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 是图像维度
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # 最大池化 (2,2) 窗口
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # 除批量维度外的所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
```

### 1.5 训练神经网络

```python
import torch.optim as optim

# 创建网络、损失函数和优化器
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 假设我们有一些训练数据
# inputs 是输入数据，labels 是标签
inputs = torch.randn(1, 1, 32, 32)
labels = torch.randn(1, 10)

# 训练循环
for epoch in range(5):  # 多次循环数据集
    # 前向传播
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    
    # 反向传播和优化
    optimizer.zero_grad()  # 清零梯度缓存
    loss.backward()        # 反向传播
    optimizer.step()       # 更新参数
    
    # 打印统计信息
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
```

### 1.6 数据加载与处理

PyTorch 提供了 torch.utils.data 模块来处理数据：

```python

from torch.utils.data import Dataset, DataLoader

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

# 创建数据集和数据加载器
dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 使用数据加载器
for batch_idx, (data, labels) in enumerate(dataloader):
    print(f'Batch {batch_idx}, Data shape: {data.shape}, Labels shape: {labels.shape}')
```

### 1.7 GPU 加速

PyTorch 支持 GPU 加速，只需将张量或模型移动到 GPU 上：

```python

# 检查 GPU 是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 将网络移动到 GPU
net.to(device)

# 将输入和目标也移动到 GPU
inputs, labels = inputs.to(device), labels.to(device)

```

### 1.8 模型保存与加载

```python
# 保存模型
PATH = './model.pth'
torch.save(net.state_dict(), PATH)

# 加载模型
net = Net()  # 必须先定义相同的网络结构
net.load_state_dict(torch.load(PATH))
net.eval()  # 设置为评估模式
```
