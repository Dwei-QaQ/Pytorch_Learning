# PyTorch_Learning

本文大量参考PyTorch官方文档和教程，但是额外加了一些例子和注释，方便理解。

以下是几个重要链接：

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

#### 1.4.6 形状广播和改变

```python

# 广播形状
x = torch.tensor([1.0, 2.0, 3.0])  # 形状 (3,)
y = torch.rand(5, 3)               # 形状 (5, 3)
print(x + y)                       # x会被广播为(5,3)

print(x.shape)                     # 添加这行查看x的形状
print(y.shape)                     # 添加这行查看y的形状

# 改变形状
a = torch.randn(4, 4)
b = a.view(16)                      # 将 a 的形状改变为 (16,)
c = a.view(-1, 8)                   # -1 表示自动计算该维度大小
print(f"{a} \n{b} \n{c} \n ")

```

输出：

```bash
tensor([[1.0405, 2.5298, 3.3739],
        [1.2869, 2.6964, 3.7740],
        [1.1309, 2.9418, 3.5148],
        [1.9281, 2.7551, 3.3660],
        [1.1602, 2.4478, 3.6667]])
//相当于把x广播为5行3列的张量

torch.Size([3])
torch.Size([5, 3])
tensor([[-0.8745,  0.5820, -0.2385, -0.7005],
        [ 0.4052,  1.9969, -0.0108,  0.9123],
        [-1.3900, -1.4591,  0.7331,  0.3289],
        [-0.4143,  1.4008,  0.1410,  1.3530]]) 
tensor([-0.8745,  0.5820, -0.2385, -0.7005,  0.4052,  1.9969, -0.0108,  0.9123,
        -1.3900, -1.4591,  0.7331,  0.3289, -0.4143,  1.4008,  0.1410,  1.3530]) 
tensor([[-0.8745,  0.5820, -0.2385, -0.7005,  0.4052,  1.9969, -0.0108,  0.9123],
        [-1.3900, -1.4591,  0.7331,  0.3289, -0.4143,  1.4008,  0.1410,  1.3530]]) 

```

### 1.5 与 NumPy 的桥接

CPU 上的张量和 NumPy 数组可以共享其底层内存位置，改变其中一个会改变另一个。

#### 1.5.1 张量转 NumPy 数组

```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

```

输出：

```bash
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]

```

张量中的改变会反映在 NumPy 数组中。

#### 1.5.2 NumPy 数组转张量

```python
n = np.ones(5)
t = torch.from_numpy(n)

```

NumPy 数组中的改变会反映在张量中。

```python
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

```

输出：

```bash
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]

```

## 2.数据集 & DataLoaders

处理数据样本的代码可能会变得杂乱且难以维护；理想情况下，我们希望将数据集代码与模型训练代码解耦，以提高可读性和模块化。PyTorch 提供了两种数据原语：torch.utils.data.DataLoader 和 torch.utils.data.Dataset，它们允许您使用预加载的数据集以及您自己的数据。Dataset 存储样本及其对应的标签，而 DataLoader 则在 Dataset 周围封装了一个迭代器，以便于访问样本。

PyTorch 领域库提供了许多预加载的数据集（例如 FashionMNIST），这些数据集继承自 torch.utils.data.Dataset 并实现了特定于特定数据的功能。它们可用于原型设计和模型基准测试。

 [图像数据集](https://pytorch.ac.cn/vision/stable/datasets.html)

 [文本数据集](https://pytorch.ac.cn/text/stable/datasets.html)

 [音频数据集](https://pytorch.ac.cn/audio/stable/datasets.html)

### 2.1 加载数据集

这是一个从 TorchVision 加载 Fashion-MNIST 数据集的示例。Fashion-MNIST 是一个包含 Zalando 商品图像的数据集，由 60,000 个训练样本和 10,000 个测试样本组成。每个样本包含一个 28×28 的灰度图像和来自 10 个类别之一的关联标签。

我们使用以下参数加载 FashionMNIST Dataset

   -root 是训练/测试数据存储的路径，

   -train 指定训练或测试数据集，

   -download=True 会在 root 路径下数据不存在时从互联网下载。

   -transform 和 target_transform 指定特征和标签变换

```python
import torch    #导入 PyTorch 深度学习框架，提供张量操作和神经网络功能
from torch.utils.data import Dataset    #导入数据集基类，用于自定义数据集
from torchvision import datasets        #导入计算机视觉专用数据集模块，包含多种预定义数据集
from torchvision.transforms import ToTensor     #导入图像转换工具，将 PIL 图像或 NumPy 数组转换为 PyTorch 张量 (Tensor)
import matplotlib.pyplot as plt     #导入 Matplotlib 绘图库，用于数据可视化


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

```

### 2.2 迭代和可视化数据集

我们可以像列表一样手动索引 Datasets：training_data[index]。我们使用 matplotlib 来可视化训练数据中的一些样本。

```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))     #创建一个 8×8 英寸的图像窗口
cols, rows = 3, 3       #设置为 3 行 3 列的网格，共显示 9 个样本
for i in range(1, cols * rows + 1):     #循环显示样本
    sample_idx = torch.randint(len(training_data), size=(1,)).item()    #生成随机索引，用于从训练集中随机选择样本
    img, label = training_data[sample_idx]      #通过索引获取图像数据和对应的标签
    figure.add_subplot(rows, cols, i)   #在网格中添加子图
    plt.title(labels_map[label])        #设置子图标题为类别名称
    plt.axis("off")     #关闭坐标轴显示
    plt.imshow(img.squeeze(), cmap="gray")      #显示图像，squeeze()用于去除维度为 1 的维度，cmap="gray"指定灰度色彩映射
plt.show()      #显示创建的图像窗口

```

输出示例：

### 2.3 为您的文件创建自定义数据集

自定义 Dataset 类必须实现三个函数：__init__、__len__ 和 __getitem__。请看这个实现；FashionMNIST 图像存储在目录 img_dir 中，而它们的标签则单独存储在 CSV 文件 annotations_file 中。

在接下来的部分，我们将详细介绍这些函数中的每一个。

```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

```

#### 2.3.1 __init__

__init__ 函数在实例化 Dataset 对象时运行一次。我们初始化包含图像的目录、标注文件以及两个变换（将在下一节详细介绍）。

labels.csv 文件如下所示

```bash
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9

```

```python
def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform

```

#### 2.3.2 __len__

__len__ 函数返回数据集中样本的数量。

```python
def __len__(self):
    return len(self.img_labels)

```

#### 2.3.3 __getitem__

__getitem__ 函数加载并返回给定索引 idx 处的数据集样本。根据索引，它确定图像在磁盘上的位置，使用 read_image 将其转换为张量，从 self.img_labels 中的 csv 数据检索相应标签，对其调用变换函数（如果适用），并以元组形式返回张量图像和相应标签。

labels.csv 文件如下所示

```python
def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = read_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return image, label

```

### 2.4 使用 DataLoaders 准备数据进行训练

Dataset 一次检索一个样本的数据集特征和标签。在训练模型时，我们通常希望以“迷你批量”的形式传递样本，在每个 epoch 重新打乱数据以减少模型过拟合，并使用 Python 的 multiprocessing 来加速数据检索。

DataLoader 是一个迭代器，它通过简单的 API 为我们抽象了这种复杂性。

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

```

### 2.5 遍历 DataLoader

我们已将数据集加载到 DataLoader 中，并可以根据需要遍历数据集。下面的每次迭代都会返回一批 train_features 和 train_labels（分别包含 batch_size=64 个特征和标签）。因为我们指定了 shuffle=True，所以在遍历所有批量后，数据会被打乱（如需对数据加载顺序进行更精细的控制，请参阅采样器 (Samplers)）。

```python
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

```

## 3.变换

数据并不总是以机器学习算法训练所需的最终处理形式出现。我们使用 变换（transforms） 对数据进行一些处理，使其适合训练。

所有 TorchVision 数据集都有两个参数 -transform 用于修改特征，target_transform 用于修改标签 - 它们接受包含变换逻辑的可调用对象。 torchvision.transforms 模块提供了几个常用的现成变换。

FashionMNIST 特征采用 PIL Image 格式，标签是整数。为了训练，我们需要将特征转换为归一化张量，将标签转换为独热编码张量。为了实现这些变换，我们使用 ToTensor 和 Lambda。

```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

```

### 3.1 ToTensor()

ToTensor 将 PIL 图像或 NumPy ndarray 转换为 FloatTensor，并将图像的像素强度值缩放到 [0., 1.] 范围内。

### 3.2 Lambda 变换

Lambda 变换应用任何用户定义的 lambda 函数。在这里，我们定义了一个函数将整数转换为独热编码张量。它首先创建一个大小为 10 的零张量（数据集中标签的数量），然后调用 scatter_ 函数，根据标签 y 给定的索引位置赋值 value=1。

```python
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

```

## 4.构建神经网络

神经网络由对数据执行操作的层/模块组成。torch.nn 命名空间提供了构建自己的神经网络所需的所有构建块。PyTorch 中的每个模块都继承自 nn.Module。神经网络本身就是一个由其他模块（层）组成的模块。这种嵌套结构使得构建和管理复杂的架构变得容易。

在接下来的部分，我们将构建一个神经网络来分类 FashionMNIST 数据集中的图像。

```python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

```

### 4.1获取用于训练的设备

我们希望能够在加速器上训练我们的模型，例如 CUDA、MPS、MTIA 或 XPU。如果当前加速器可用，我们将使用它。否则，我们使用 CPU。

### 神经网络 (Neural Networks)基础

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

### 训练神经网络

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

### GPU 加速

PyTorch 支持 GPU 加速，只需将张量或模型移动到 GPU 上：

```python

# 检查 GPU 是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 将网络移动到 GPU
net.to(device)

# 将输入和目标也移动到 GPU
inputs, labels = inputs.to(device), labels.to(device)

```

### 模型保存与加载

```python
# 保存模型
PATH = './model.pth'
torch.save(net.state_dict(), PATH)

# 加载模型
net = Net()  # 必须先定义相同的网络结构
net.load_state_dict(torch.load(PATH))
net.eval()  # 设置为评估模式
```
