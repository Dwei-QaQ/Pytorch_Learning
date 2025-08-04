import torch
import numpy as np

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(f"Tensor: \n {x_data} \n")

np_array = np.array(data)
x_np = torch.from_numpy(np_array)


x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")


tensor = torch.ones(4, 4)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

t2 = torch.stack([tensor, tensor, tensor], dim=1)
print(t2)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print(f"{y1} \n{y2} \n{y3} \n ")

z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(f"{z1} \n{z2} \n{z3} \n ")

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

x = torch.tensor([1.0, 2.0, 3.0])  
y = torch.rand(5, 3)               
print(x + y)  

print(x.shape)  
print(y.shape)  

a = torch.randn(4, 4)
b = a.view(16)                      
c = a.view(-1, 8)                   
print(f"{a} \n{b} \n{c} \n ")

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

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
