# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:27:06 2021

@author: cemsi
"""

#3-2 Linear Regression Implementation from Scratch
# Many frameworks already make calculations. But, it is beneficial to learn the implementation because 
#deeper changes can be made. Only etnsors and auto differentiation will be used from framework.

import random
import torch
from pytorch_d2l.d2l import torch as d2l

#Creating artificial dataset with noise.

def synthetic_data(w, b, num_examples): #@save
    """ generate y = Xw + b + € (noise) """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

"""
print('features:', features[0], '\nlabel:', labels[0])

#Generating scatter plot 
d2l.set_figsize()
d2l.plt.scatter(features[:,(1)].detach().numpy(), labels.detach().numpy(), 1)
"""

#Iterating for mini-batches

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
        
        
        
#Setting batch size an arbitrary number
batch_size = 10

"""        
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
"""       
 
#Initializing random weights and biases by random sampling
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

        
#Defining a model: To calculate the output of linear model, matrix-vector dot product will be made.
#Input features X, model weights w, biases (offsets) b to each example. 

def linreg(X, w, b): #@save
    """ linear regression model. """
    return torch.matmul(X, w) + b


#Defining loss function: Gradient of loss function as squared loss function. y_hat is the prediction.

def squared_loss(y_hat, y): #@save
     return (y_hat - y.reshape(y_hat.shape))**2 / 2
 
    
#Defining the optimization algorithm: Minibatch stochastic gradient descent????

def sgd(params, lr, batch_size): #@save
    """Minibatch stochastic gradient descent """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
            
    
#Training: In each iteration, a minibatch of training examples will be grabbed and passed through the model
#to obtain a set of predictions. After calculating the loss, the backward will be initiated through the network
#storing the gradients with respect to each parameter. Finally optimization algorithm sgd() will be called to 
#update the parameters.
#number of epochs and learning rates are hyperparamters and optimization of these are hard to tell.
"""
The loop is:
    1-initialize parameters(w, b)
    2-Repeat until done
        a-compute gradient 
        b-update parameters
"""
    
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        #Mini-batch loss function 
        l = loss(net(X, w, b), y)
        #Computing gradient on l
        l.sum().backward()
        #Update parameters 
        sgd([w, b], lr, batch_size)
        
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
        
    
        
        
    
    
    
    
    
    
    
    
    
    
    