# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 18:28:44 2021

@author: cemsi
"""

#2-6 Probability
from matplotlib import pyplot as plt
from torch.distributions import multinomial
import torch
from pytorch_d2l.d2l import torch as d2l
#Sampling: Drwaing examples from probability distributions.
#Multinomial distribution: The distribution that assigns probabilities to a number of discrete choices.

fair_probs = torch.ones([6])/6
#print(multinomial.Multinomial(10, fair_probs).sample())

#Store the results
counts = multinomial.Multinomial(1000, fair_probs).sample()
print(counts/1000)

#Visualize the sampling
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts/cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6,4.5))
for i in range(6):
    d2l.plt.plot(estimates[:,i].numpy(), label=("P(die=)" + str(i+1) + ")"))
    
d2l.plt.axhline(y=0.167, color='black', linestyle = 'dashed')
d2l.plt.gca().set_xlabel("Groups of experiments")
d2l.plt.gca().set_ylabel("Estimated probability")
d2l.plt.legend()

#Sample space or outcome space: the set of S where each element is an outcome.
#Event: A set of outcomes from a given sample space. 


