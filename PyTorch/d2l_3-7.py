#!/usr/bin/env python
# coding: utf-8

# In[3]:


#3-7 Concise Implementation of Softmax Regression, this will be a high-level implementation of previous Softmax Regression.

import torch
from torch import nn
from pytorch_d2l.d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        
net.apply(init_weights)
#Loss function
loss = nn.CrossEntropyLoss()
#Optimization (stochastic)
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
#Training
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

"""
Summary:
    Using high-level APIs, we can implement softmax regression much concisely.
    From a computational perspective, implementing softmax regression has intricacies. In many cases, a deep learning
framework takes additional precautions beyond these most well-known tricjs to ensure numerical stability, saving us form even
more pitfalls that we would encounter if we tried to code all of our models from scratch in practice.
"""


# In[ ]:





# In[ ]:




