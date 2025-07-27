import torch

# 正确示例1：相同形状
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print(x + y)  # 可以正常相加

# 正确示例2：可广播的形状
x = torch.tensor([1.0, 2.0, 3.0])  # 形状 (3,)
y = torch.rand(5, 3)               # 形状 (5, 3)
print(x + y)  # x会被广播为(5,3)

#print(x.shape)  # 添加这行查看x的形状
#print(y.shape)  # 添加这行查看y的形状
#print(x + y)

x = torch.randn(4, 4)
y = x.view(16)  # 展平
z = x.view(-1, 8)  # -1 表示自动计算该维度大小
print(x, y, z)