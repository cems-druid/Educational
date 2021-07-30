# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 19:08:58 2021

@author: cemsi
"""

#3-1 Linear Regression:
#Regression: A set of methods for modelling the relationsjip between one or more independent
#variables and a dependent variable, it characterizes the relationship between the inputs and outputs.

"""
Gradient descent: The key technique for optimizing most of the deep learning models consist of 
iteratively reducing the error by updating the parameters in the direction that incrementally 
lowers the loss function. This practice is very slow, in order to update one iteration
it needs to pass over entire dataset. 

Minibatch stochastic gradient descent: Since it is very slow to pass over all the dataset,
alternatively it can pass over some random sampling over the dataset.

"""
import math
import time
import numpy as np
import torch
from pytorch_d2l.d2l import torch as d2l

n = 10000
a = torch.ones(n)
b = torch.ones(n)

class Timer: #@save
    """record multiple running times"""

    def __init__(self):
        self.times = []
        self.start()
        
    def start(self):
        """start the timer"""
        self.tik = time.time()
        
    def stop(self):
        """stop the timer and record the time in a list"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def avg(self):
        """return the average time"""
        return sum(self.times) / len(self.times)
    
    def sum(self):
        """return the sum of time"""
        return sum(self.times)
    
    def cumsum(self):
        """return the accumulated time."""
        return np.array(self.array).cumsum().tolist()
    
"""    
c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
        
print(f'{timer.stop(): .5f} sec')
#Or sum can be done in element-wise fashion, which makes it faster since code is vectorized.

timer.start()
d = a + b
print(f'{timer.stop():.5f} sec')
"""

#Normal distribution and squared loss function
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2*math.pi*sigma**2)
    return p*np.exp(-0.5/sigma**2*(x-mu)**2)

x = np.arange(-7,7,0.01)
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x,mu,sigma) for mu, sigma in params], xlabel='x', ylabel='p(x)',
             figsize=(4.5, 2.5), legend=[f'mean {mu}, std {sigma}' for mu,sigma in params])



        
        
        
        
