#!/usr/bin/env python
# coding: utf-8

# In[1]:


#4-1 Multilayer Perceptrons

get_ipython().run_line_magic('matplotlib', 'inline')
import torch
from pytorch_d2l.d2l import torch as d2l

"""
Summary:
    Multi layer perceptron (MLP) adds one or multiple fully-connected hidden layers between the output 
and input layers and transforms the output of the hidden layer via an activation function.
    Commonly-used activation functions include the ReLU function, the sigmoid function and the tanh function.
"""

#Activation functions: rectified linear unit (ReLU) : ReLU(x) = max(x,0)

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5,2.5))


# In[3]:


#Derivative plot of ReLU
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5,2.5))


# In[4]:


#Sigmoid function: sigmoid(x)=1/1+exp(-x) --> range:(0,1)
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5,2.5))


# In[ ]:


#Derivative of sigmoid function's plot:
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)


# In[6]:


#Tanh function: tanh(x) = 1-exp(-2x)/1+exp(-2x)
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5,2.5))


# In[7]:


#Derivative of tanh activation function's plot
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))

