import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

'''
PyTorch fundamentals walkthrough 
'''
#scalar (0 dim tensor)
scalar = torch.tensor(7)
scalar.item()

#Vector (1 dim)
vector = torch.tensor([2,3])

#Matrix (2 dim)
MATRIX = torch.tensor([[4,5], [5,6]])


#Tensor (any dim)
TENSOR = torch.tensor([[[1,2,3], [3,5,8]]
                      ])

#Random tesnors (super useful in ML workflow)

random_tensor = torch.rand(2,1,2,2)

print(random_tensor)

random_image = torch.rand(size=(224,224,3)) # height, width, colour channels (RGB)


#Zeros and ones

zeros = torch.zeros(3,4)
print(zeros)

ones = torch.ones(3,4)
ones.dtype #check the type, default is float32


#Range of tensors and tensors-like

ranged = torch.arange(1,10,2)
print(ranged)

#tensor-like
ten_zeros = torch.zeros_like(input = ranged)

print(ten_zeros)

#Tensor datatypes 
type_test_tensor = torch.tensor([3.0, 6.0, 9.0], 
                                dtype = None, 
                                device="cuda", 
                                requires_grad=False) #whether or not to track gradients with this tensor's operations

float16_tensor = type_test_tensor.type(torch.float16)

#Attributes
print(f"Datatype: {float16_tensor.dtype}")
print(f"Shape: {float16_tensor.shape}")
print(f"Device: {float16_tensor.device}")


#Tensor operations - matrix multiplication
mmm_test = torch.matmul(type_test_tensor, type_test_tensor) #or torch.mm 
print(mmm_test)


#Transpoe 
tens_1 = torch.rand(2,3)
t_tens1 = tens_1.T

print(t_tens1)

#Tensor aggregation (going from large amount of numbers to small)

tens_1.min
tens_1.max
tens_1.mean


#Positional min/max 

print(tens_1.argmax())
print(tens_1.argmin())

#Reshaping, stcking, squeezing, and unsqeezingtensors

'''
------------------------------------------------
Reashaping - input tensor to a defined shape

View - return a view of an input tensor of certain shape but keep the same memory as original tensor 

Stacking- combine multiple tensors on top of each other (torch.vstack) or side by side (torch.hstack)

Squeeze - removes all '1' dimensions

Unsqueeze - adds a '1' dimension to a target tensor 

Permute - Return a view of the input with dimensions permuted (swapped) in a certain way 

--------------------------------------------- 

'''

x = torch.arange(1,10)
print(x)

#Shape
x_reshaped = x.reshape(1,9) #Reshape has to be compatible with original size 

print(x_reshaped)

x_reshaped = x.reshape(9,1)

print(x_reshaped)

#View
z=x.view(1,9) #Changing z changes x, because they share memory

print(z)

z[:,0]=5
print(f"x: {x} z:{z}")

#Stack tensors on top of each other

x_stacked = torch.vstack([x,x])
print(x_stacked)

#Squeezing
x = torch.zeros(2, 1, 2, 1, 2)
print(x.size())
y = torch.squeeze(x)
print(y.size())
y = torch.squeeze(x, 0) #specified the ceratin location - no single dim - no change
print(y.size())
y = torch.squeeze(x, 1)
y.size()
y = torch.squeeze(x, (1, 2, 3))

y = torch.unsqueeze(x,1)
print(y.size())


#Permute
x = torch.zeros(1,2,3)
print(x.size())
x_permuted = torch.permute(x, (2,1,0))
print(x_permuted.size())

#Indexing
x= torch.arange(1,10).reshape(1,3,3)

print(x, x.shape)
print(x[0,2,2])

#: gets all of a target dimension 
print(x[:, :,2])



#PyTorch tensors and NumPy

#Note: deafult numpy dtype is float64

array = np.arange(1,8)

tensor = torch.from_numpy(array)

print(array, tensor)

tensor = torch.ones(7)
numpy_tensor = tensor.numpy()

print(tensor, numpy_tensor)

#Reproducibility (trying to take random out of random)

'''
To reduce the randomness random seed is used - seed flavours
'''
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3,4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3,4)

print(random_tensor_C==random_tensor_D)


#PyTorch + CUDA

device = "cuda" if torch.cuda.is_available else "cpu"

tensor = torch.tensor([1,2.3])
gpu_tensor = tensor.to(device)
print(gpu_tensor, gpu_tensor.device)

#If tensor is on GPu can't convert it to NumPy

tensor_back = gpu_tensor.cpu().numpy()
