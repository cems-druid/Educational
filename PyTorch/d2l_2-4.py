# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:48:36 2021

@author: cemsi
"""

#2-4 Calculus:
    
#Differentiation example of 3x^2-4x
#%matplotlib inline
import numpy as np
from IPython import display
from matplotlib import pyplot as plt
#from d2l import torch as d2l

def f(x):
    return 3*x**2 - 4*x

def numerical_lim(f, x, h):
    return (f(x+h) - f(x))/h

h = 0.1

"""
for i in range(5):
    print(f'h={h:.5f}, numerical limit = {numerical_lim(f,1,h):.5f}')
    h *= 0.1
"""
  
def use_svg_display(): #@save
    """ use the svg format to display a plot in Jupyter"""
    
    display.set_matplotlib_formats('svg')
    
def set_figsize(figsize=(3.5, 2.5)): #@save
    """Set the figure size for matplotlib """
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize
    
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """set axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
    
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear',
         yscale='linear', fmts=('-','m--', 'g-', 'r:'), figsize=(3.5,2.5), axes=None):
    """plot data points"""
    if legend is None:
        legend = []
        
    set_figsize(figsize)
    axes = axes if axes else plt.gca()
    
    #return true if X is list or tensor with one dimension (1 axis)
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__"))
    
    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else: 
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2*x-3], 'x', 'f(x)', legend=['f(x)','Tangent line (x=1)'])

    














