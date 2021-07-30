# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 17:34:03 2021

@author: cemsi
"""

#3-3 Concise implementation of linear regression: In this section framework's built-up functions will be used.
#This is higher level API usage.

#Creating a random dataset.

import numpy as np
import torch
from torch.utils import data
from pytorch_d2l.d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

#Iterating over the dataset, is_train: do we want to shuffle the data on each epoch 

def load_array(data_arrays, batch_size, is_train=True): #@save
    """ construct a pytorch data iterator"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

#Check if data_iter works 
#print(next(iter(data_iter)))

#Define the model: linear regression will be used
#The sequential class defines a container for several layers that will be chained together.

from torch import nn
net = nn.Sequential(nn.Linear(2, 1))

#model parameters, setting the random sampling from normal distribution with mean=0 and 
#standard deviation = 0.01 and bias = 0

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

#loss function as mean squared error
loss = nn.MSELoss()

#Optimization algorithm as minibatch stochastic gradient descent
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

#training: reminder as what is being done
"""
For some number of epochs, dataset will be passed over completely, grabbing one minibatch of inputs and the
corresponding ground-truth labels. For each minibatch:
    Generating predictions by calling net(X) and calculating the loss "l" (this is forward propagation)
    Calculating gradients by running backpropagation
    Updating the model parameters by invoking the optimizer
"""

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f"epoch {epoch+1}, loss {l:f}")
    
    

#comparing with original ones

w = net[0].weight.data
print("error in estimating w: ", true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print("error in estimating b: ", true_b - b)

"""
Summary:
    -Using PyToch's high level APIs, we can implement models much more concisely.
    -In PyTorch, the data module provides tools for data processing, the nn module defines a large number of 
    neural network layers and common loss functions
    -We can initialize the parameters by replacing their values with methods ending with _.
"""

