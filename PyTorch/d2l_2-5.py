# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 17:37:11 2021

@author: cemsi
"""

#2-5: Automatic Differentiation
#Starting from a toy example, differantiating the function y = 2x(T)x with respect to column vector x.
import torch

x = torch.arange(4.0)

#We always try to lower the consumption of new memory.
#Note: A gradient of a scalar-valued function with respect to a vector x is itself vector-valued and has the same shape as x.

x.requires_grad_(True)
x.grad

#Calculating y's gradient automatically with respect to each component of x by backpropagation 
y = 2*torch.dot(x,x)
y.backward()
x.grad

#PyTorch accumulates the gradient in default, it needs to be clean the values everytime
x.grad.zero_()
y = x.sum()
y.backward()
