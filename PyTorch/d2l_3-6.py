#!/usr/bin/env python
# coding: utf-8

# In[27]:


#3-6: Implementation of Softmax Regression from Scratch

import torch
from IPython import display
from pytorch_d2l.d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#Pictures are 28x28, to use them they will be transformed into 784 length vectors
num_inputs = 784
num_outputs = 10

#In softmax regression there are as many outputs as there are classes. 
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

#Softmax has three steps: i) exponentiate each term, ii)sum each row to get normalizaion constant 
#iii)divide each row by normalizaion constat, denominator or normalization constant is also called partition function.
#softmax(X)[i][j] = (exp(X)[i][j]) / sum(k)(exp(X[i][k]))
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition 


X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
#Checking to see the probabilities
#print(X_prob, X_prob.sum(1))

#Defining a model, X turned into a vector
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W)+b)

#Defining a loss function, in this case a cross-entropy loss function. 
#Cross-entropy takes the negative log-likelihood of the predicted probability assigned to the true label. 
#The code below is used rather than iterating over loop. y_hat is predicted probability distribution.

y = torch.tensor([0,2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
#print(y_hat[[0, 1], y])

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

#print(cross_entropy(y_hat, y))

#Accuracy calculation
def accuracy(y_hat, y): #@save
    """
    Compute the number of correct predictions
    y: outputs
    y_hat: predictions
    """
    #Assumption is if y_hat is matrix, the second dimension stores the predictions.
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
        
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

#print(accuracy(y_hat,y)/len(y))

def evaluate_accuracy(net, data_iter): #@save
    """
    Compute the accuracy for a model on a dataset.
    """
    #Change into evaluation mode
    if isinstance(net, torch.nn.Module):
        net.eval()
        
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
        
    return metric[0]/metric[1]
    

#Utility class that helps with storing number of correct predictions and number of predictions.
class Accumulator: #@save
    """For accumulating sums over n variables"""
    
    def __init__(self, n):
        self.data = [0,0]*n
        
    def add(self, *args):
        self.data = [a+float(b) for a,b in zip(self.data, args)]
        
    def reset(self):
        self.data = [0,0]*len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
    
#Net has random samples inside of it
#print(evaluate_accuracy(net, test_iter))

#Training
def train_epoch_ch3(net, train_iter, loss, updater): #@save
    #Set net to training mode
    if isinstance(net, torch.nn.Module):
        net.train()
        
    metric = Accumulator(3)
    for X, y in train_iter:
        #Computing gradients and updating parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        
        #Use torch's built-up optimizer
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(float(l)*len(y), accuracy(y_hat, y), y.numel())
        #Use custom optimizer
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
            
    #Returning training loss, training accuracy
    return metric[0]/metric[2], metric[1]/metric[2]


#Another utility class as plot data in animation.
class Animator: #@save
    """For plotting data in animation"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
                fmts=('-','m--','g--','r:'), nrows=1, ncols=1, figsize=(3.5, 2.5)):      
        if legend is None:
            legend=[]
            
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows*ncols == 1:
            self.axes = [self.axes, ]
            
        self.config_axes = lambda: d2l.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        
    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
            
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x]*n
            
        if not self.X:
            self.X = [[] for _ in range(n)]
            
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        
        for i, (a,b) in enumerate(zip(x,y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        
        self.axes[0].cla()
        
        for x,y,fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
            
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
        
        
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater): #@save
    
    animator = Animator(xlabel="epoch", xlim=[1, num_epochs], ylim=[0.3, 0.9], legend=['train loss', 'train acc', 'test acc'])
    
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch+1, train_metrics+(test_acc,))
    
    train_loss, train_acc = train_metrics
    
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
    
#Using minibatch stochastic gradient descet to optimize the loss function with learning rate 0.1
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10
#train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

#Prediction
def predict_ch3(net, test_iter, n=6): #@save
    
    for X, y in test_iter:
        break
        
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n,28,28)),1,n, titles=titles[0:n])
    
predict_ch3(net, test_iter)

"""
Summary:
    -With softmax regression, we can train models for multiclass classification.
    -The training loop of softmax regression is very similar to that in linear regression: retrieve and read data, define models 
    and loss functions, train models using optimization algorithms. Most common deep learning models have similar training procedures.
"""


# In[20]:


y_hat

