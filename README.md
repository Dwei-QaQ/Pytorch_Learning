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

自定义 Dataset 类必须实现三个函数：\_\_init__、\_\_len__ 和 \_\_getitem__。请看这个实现；FashionMNIST 图像存储在目录 img_dir 中，而它们的标签则单独存储在 CSV 文件 annotations_file 中。

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

#### 2.3.1 \_\_init__

\_\_init__ 函数在实例化 Dataset 对象时运行一次。我们初始化包含图像的目录、标注文件以及两个变换（将在下一节详细介绍）。

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

#### 2.3.2 \_\_len__

\_\_len__ 函数返回数据集中样本的数量。

```python
def __len__(self):
    return len(self.img_labels)

```

#### 2.3.3 \_\_getitem__

\_\_getitem__ 函数加载并返回给定索引 idx 处的数据集样本。根据索引，它确定图像在磁盘上的位置，使用 read_image 将其转换为张量，从 self.img_labels 中的 csv 数据检索相应标签，对其调用变换函数（如果适用），并以元组形式返回张量图像和相应标签。

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

### 4.1 获取用于训练的设备

我们希望能够在加速器上训练我们的模型，例如 CUDA、MPS、MTIA 或 XPU。如果当前加速器可用，我们将使用它。否则，我们使用 CPU。

```python
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
```

在配置了cuda环境后可以显示：

```bash
Using cuda device
```

### 4.2 定义类

我们通过继承 nn.Module 来定义我们的神经网络，并在 \_\_ init__ 中初始化神经网络层。每个 nn.Module 子类都在 forward 方法中实现对输入数据的操作。

```python
class NeuralNetwork(nn.Module): #定义了名为NeuralNetwork的类，继承自 PyTorch 的nn.Module基类
    def __init__(self):
        super().__init__()  # 初始化父类nn.Module
        self.flatten = nn.Flatten()  # 展平层定义
        self.linear_relu_stack = nn.Sequential(  # 序列式网络结构
            nn.Linear(28*28, 512),  # 输入层：28×28=784维 → 512维
            nn.ReLU(),              # ReLU激活函数
            nn.Linear(512, 512),    # 隐藏层：512维 → 512维
            nn.ReLU(),              # ReLU激活函数
            nn.Linear(512, 10),     # 输出层：512维 → 10维
    )

    def forward(self, x):
    x = self.flatten(x)          # 展平输入张量
    logits = self.linear_relu_stack(x)  # 通过全连接网络
    return logits                # 返回原始输出分数(未经过softmax)
```

我们创建 NeuralNetwork 的一个实例，并将其移动到 device，然后打印其结构。

```python
model = NeuralNetwork().to(device)
print(model)
```

输出：

```bash
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```

要使用模型，我们将输入数据传递给它。这将执行模型的 forward 方法，以及一些后台操作。不要直接调用 model.forward()！

在输入上调用模型会返回一个二维张量，其中 dim=0 对应于每个类别的 10 个原始预测值，dim=1 对应于每个输出的单个值。通过将其传递给 nn.Softmax 模块的一个实例，我们可以获得预测概率。dim 参数表示值必须沿哪个维度求和为 1。

```python
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

输出：

```bash
Predicted class: tensor([6], device='cuda:0')
```

### 4.3 模型层

让我们分解 FashionMNIST 模型中的层。为了说明这一点，我们将取一个 3 张 28x28 尺寸图像的样本小批量，并查看它通过网络时发生的变化。

```python
input_image = torch.rand(3,28,28)
print(input_image.size())
```

输出：

```bash
torch.Size([3, 28, 28])
```

#### 4.3.1 nn.Flatten

我们初始化 nn.Flatten 层，将每张 2D 28x28 图像转换为包含 784 个像素值的连续数组（小批量维度（dim=0）被保留）。

```python
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
```

输出：

```bash
torch.Size([3, 784])
```

#### 4.3.2 nn.Linear

线性层是一个使用其存储的权重和偏置对输入应用线性变换的模块。

```python
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
```

输出：

```bash
torch.Size([3, 20])
```

#### 4.3.3 nn.ReLU

非线性激活层在模型的输入和输出之间创建复杂的映射。它们在线性变换之后应用，以引入非线性，帮助神经网络学习各种现象。

在此模型中，我们在线性层之间使用 nn.ReLU，但还有其他激活函数可以在模型中引入非线性。

```python
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
```

```bash
Before ReLU: tensor([[ 0.2901, -0.0419, -0.4242,  0.4214, -0.0256,  0.1765, -0.0505,  0.0774,
          0.1546, -0.3323,  0.2471,  0.0810,  0.0872, -0.2698, -0.1555,  0.2408,
          0.2148, -0.0473,  0.1200,  0.3072],
        [ 0.4119, -0.1023, -0.1870,  0.3657, -0.1041,  0.1881,  0.2143,  0.5162,
          0.3975, -0.4295,  0.1336,  0.0949,  0.0736, -0.1902, -0.5618,  0.2341,
          0.1901, -0.1224,  0.2775,  0.5580],
        [ 0.4820,  0.2787, -0.3580,  0.5412,  0.1555,  0.2166,  0.0193,  0.2851,
          0.5649, -0.0321,  0.0851, -0.1493,  0.0353, -0.1425, -0.4309,  0.0231,
         -0.1037, -0.0754, -0.0939,  0.3784]], grad_fn=<AddmmBackward0>)


After ReLU: tensor([[0.2901, 0.0000, 0.0000, 0.4214, 0.0000, 0.1765, 0.0000, 0.0774, 0.1546,
         0.0000, 0.2471, 0.0810, 0.0872, 0.0000, 0.0000, 0.2408, 0.2148, 0.0000,
         0.1200, 0.3072],
        [0.4119, 0.0000, 0.0000, 0.3657, 0.0000, 0.1881, 0.2143, 0.5162, 0.3975,
         0.0000, 0.1336, 0.0949, 0.0736, 0.0000, 0.0000, 0.2341, 0.1901, 0.0000,
         0.2775, 0.5580],
        [0.4820, 0.2787, 0.0000, 0.5412, 0.1555, 0.2166, 0.0193, 0.2851, 0.5649,
         0.0000, 0.0851, 0.0000, 0.0353, 0.0000, 0.0000, 0.0231, 0.0000, 0.0000,
         0.0000, 0.3784]], grad_fn=<ReluBackward0>)
```

#### 4.3.4 nn.Sequential

nn.Sequential 是模块的有序容器。数据按定义的相同顺序通过所有模块。可以使用顺序容器快速构建网络，例如 seq_modules。

```python
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
```

#### 4.3.5 nn.Softmax

神经网络的最后一个线性层返回 logits - 位于 [-infty, infty] 的原始值 - 然后传递给 nn.Softmax 模块。Logits 被缩放到 [0, 1] 的值，表示模型对每个类别的预测概率。dim 参数指示值必须沿哪个维度求和为 1。

```python
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
```

### 4.4 模型参数

神经网络内部的许多层都是\*参数化\*的，即拥有在训练期间优化的相关权重和偏置。继承 nn.Module 会自动跟踪模型对象内部定义的所有字段，并通过模型的 parameters() 或 named_parameters() 方法使所有参数可访问。

在此示例中，我们遍历每个参数，并打印其大小和值的预览。

```python
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```

```bash
Model structure: NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)


Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[ 0.0086, -0.0113,  0.0069,  ..., -0.0022, -0.0021,  0.0081],
        [ 0.0065, -0.0093, -0.0350,  ...,  0.0059,  0.0218,  0.0346]],
       device='cuda:0', grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([-0.0094, -0.0320], device='cuda:0', grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[-0.0229,  0.0003, -0.0418,  ...,  0.0155, -0.0376,  0.0316],
        [-0.0135, -0.0401,  0.0377,  ..., -0.0144,  0.0202,  0.0073]],
       device='cuda:0', grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([-0.0386,  0.0300], device='cuda:0', grad_fn=<SliceBackward0>) 

Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[ 0.0190,  0.0182, -0.0429,  ..., -0.0202,  0.0352, -0.0211],
        [-0.0437, -0.0164,  0.0374,  ...,  0.0201, -0.0381,  0.0375]],
       device='cuda:0', grad_fn=<SliceBackward0>) 
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
