# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:26:50 2021

@author: cemsi
"""

#2.1.1 Getting Started

import torch

x = torch.arange(12)
x.shape
x.numel()

x1 = x.reshape(3,4)

#Zero torch.zeros((2,3,4))
#Ones torch.ones((2,3,4))
#Random torch.randn(3,4)

#Element-wise operations x+y, x-y, x*y, x/y, x**y

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])

#Exponentiation torch.exp(x)

#Concatenation multiple tensors.
#Axis 0 = rows, Axis 1 = columns 

x = torch.arange(12, dtype=torch.float32).reshape((3,4))
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
#torch.cat((x, y), dim=0), torch.cat((x, y), dim=1)

#Binary operation, x == y
#Summation, x.sum()

#If two different shaped matrices wanted to be binary operated, it broadcasts in order to match
#the matrices, regarding the shapes.

a = torch.arange(3).reshape((3,1))
b = torch.arange(2).reshape((1,2))

#a,b 
#a+b

#x[-1] selects last element, x[1:3] selects second to third element
#x[0:2, :] indexes first and second rows and all columns

#Saving memory, typically updates will be done "in place". 
before = id(y)
y = y + x
#they are not same check, id(y) == before
#To perform in place operations easily
z = torch.zeros_like(y)
#print('id(z): ', id(z))
z[:] = x + y
#print('id(z): ', id(z))

#Conversion to other python objects
a = x.numpy()
b = torch.tensor(a)
print(type(a), type(b))







