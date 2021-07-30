#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from pytorch_d2l.d2l import torch as d2l

"""
Summary:
    Implementing MLP is not hard.
    With large number of layers, implementing MLPs from scratch can still get messy
"""

#4-2 Implementation of Multilayer Perceptrons from Scratch
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#This mlp will have 1 hidden layer with 256 hidden units. Note: It is nicer for computers to calculate powers of 2.
#For every layer weight and bias vectors need to be held.

num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True)*0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))

W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True)*0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

#ReLU activation function 
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X,a)

#Model, reshaping 2D image into vector. @ is matrix multiplication in this context.
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)
    return (H @ W2 + b2)

#Loss function
loss = nn.CrossEntropyLoss()

#Training is the same as softmax regression which was made before.
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)


# In[2]:


#Evaluation:
d2l.predict_ch3(net, test_iter)

