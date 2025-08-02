# PyTorch_Learning

官方教程: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)

PyTorch 文档: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

本文的基础是已经完成了 PyTorch 的安装以及其他环境的配置，安装教程请参考 [PyTorch 安装](https://pytorch.org/get-started/locally/)。

遇到不会的操作可以多参考官方文档和教程，同时可以在vscode里直接查询相关函数定义与用法。

欢迎与作者交流沟通！WELCOME TO COMMUNICATE WITH THE AUTHOR!

## 1.PyTorch 基础学习内容

PyTorch 是一个开源的 Python 机器学习库，基于 Torch，用于自然语言处理等应用程序。以下是 PyTorch 的基础学习内容：

### 1.1 张量基础概念

   -张量 (Tensors): PyTorch 的基本数据结构，与数组和矩阵非常相似，类似于 NumPy 的 ndarray，但可以在 GPU 上运行,张量可以看作是多维数组，其维度（dim）称为轴（axis）或阶（rank），在 PyTorch 中，我们使用张量来编码模型的输入和输出，以及模型的参数。实际上，张量和 NumPy 数组通常可以共享底层内存，从而无需复制数据。

   -自动微分 (Autograd): PyTorch 的自动微分引擎，用于神经网络训练

   -计算图 (Computational Graph): PyTorch 使用动态计算图，也称为 define-by-run 框架

```python
import torch
import numpy as np

```

### 1.2 初始化张量

张量可以通过多种方式进行初始化。

#### 1.2.1 直接从数据创建

张量可以直接从数据创建。数据类型会自动推断。

```python
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(f"Tensor: \n {x_data} \n")

```

输出：

```bash
Tensor: 
 tensor([[1, 2],
        [3, 4]]) 

```

#### 1.2.2 从 NumPy 数组创建

张量可以从 NumPy 数组创建（反之亦然）。

```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

```

#### 1.2.3 从另一个张量创建

新张量会保留参数张量的属性（形状、数据类型），除非显式覆盖。

```python
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

```

输出：

```bash
Ones Tensor:
 tensor([[1, 1],
        [1, 1]])
//强制转换为每一位为1的张量

Random Tensor: 
 tensor([[0.7812, 0.1073],
        [0.0350, 0.0997]]) 
//强制转换为每一位为随机浮点数的张量

```

#### 1.2.4 使用随机值或常量值创建

shape 是一个张量维度的元组。在下面的函数中，它决定了输出张量的维度。

```python
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

```

输出：

```bash
//shape = (2,3,)输出两行三列的张量
Random Tensor: 
 tensor([[0.6068, 0.9242, 0.4634],
        [0.6015, 0.0042, 0.9946]]) 

Ones Tensor: 
 tensor([[1., 1., 1.],
        [1., 1., 1.]]) 

Zeros Tensor: 
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

### 1.3 张量的属性

张量属性描述了它们的形状、数据类型以及存储它们的设备。

```python
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

输出：

```bash
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

### 1.4 张量操作

超过 1200 种张量操作，包括算术、线性代数、矩阵操作（转置、索引、切片）、采样等等，都在[此页面](https://pytorch.ac.cn/docs/stable/torch.html)进行了全面描述。

这些操作都可以在 CPU 和 加速器上运行，例如 CUDA、MPS、MTIA 或 XPU。如果您使用 Colab，可以通过前往“运行时”>“更改运行时类型”>“GPU”来分配一个加速器。

默认情况下，张量在 CPU 上创建。我们需要使用 .to 方法（在检查加速器可用性后）将张量显式移动到加速器上。请记住，在设备之间复制大型张量可能会消耗大量时间和内存！

```python
# We move our tensor to the current accelerator if available
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
    
```

#### 1.4.1 标准的 NumPy 式索引和切片

```python
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

```

输出：

```bash
First row:  tensor([1., 1., 1., 1.])
First column:  tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
        
```

#### 1.4.2 连接张量

可以使用 torch.cat 沿着给定维度连接一系列张量。

```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

```

输出：

```bash
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])

```

```bash
官方文档
Example:

    >>> x = torch.randn(2, 3)
    >>> x
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> torch.cat((x, x, x), 0)
    tensor([[ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497],
            [ 0.6580, -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497]])
    >>> torch.cat((x, x, x), 1)
    tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
             -1.0969, -0.4614],
            [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
             -0.5790,  0.1497]])

```

torch.stack连接

```python
t2 = torch.stack([tensor, tensor, tensor], dim=1)
print(t2)

```

输出：

```bash
tensor([[[1., 0., 1., 1.],
         [1., 0., 1., 1.],
         [1., 0., 1., 1.]],

        [[1., 0., 1., 1.],
         [1., 0., 1., 1.],
         [1., 0., 1., 1.]],

        [[1., 0., 1., 1.],
         [1., 0., 1., 1.],
         [1., 0., 1., 1.]],

        [[1., 0., 1., 1.],
         [1., 0., 1., 1.],
         [1., 0., 1., 1.]]])
//与torch.cat相比这个会增加维度

```

#### 1.4.3 算数运算

```python
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print(y3)
print(f"{y1} \n{y2} \n{y3} \n ")

# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(f"{z1} \n{z2} \n{z3} \n ")

```

输出：

```bash
tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]]) 
tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]]) 
tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]]) 
 
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]]) 
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]]) 
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]]) 

```

#### 1.4.4 单元素张量

如果您有一个单元素张量，例如通过将张量的所有值聚合到一个值中获得，您可以使用 item() 将其转换为 Python 数值。

```python
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

```

输出：

```bash
12.0 <class 'float'>

```

#### 1.4.5 就地操作

将结果存储到操作数中的操作称为就地操作。它们以_后缀表示。例如：x.copy_(y)、x.t_() 会改变 x。

```python
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

```

输出：

```bash
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])

```

就地操作节省了一些内存，但在计算导数时可能会有问题，因为会立即丢失历史记录。因此，不建议使用它们。

#### 1.4.5 形状广播

```python

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

### 1.5  自动微分 (Autograd)

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

### 1.6 神经网络 (Neural Networks)基础

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
