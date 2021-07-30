#!/usr/bin/env python
# coding: utf-8

# In[4]:


#3-5 Image Classification Dataset

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from pytorch_d2l.d2l import torch as d2l

d2l.use_svg_display()

#Download the MNIST dataset
#transforms change images from PIL type to 32-bit tensors.
trans = transforms.ToTensor()
#train set 10 categories 6000 each = 60,000 total
#test set 10 categories 1000 each = 10,000 total
mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)

#Checking data
#print(len(mnist_train), len(mnist_test))
#print(mnist_train[0][0].shape)

#dataset is about fashion or clothes, categories are tshirt, trousers, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot.
#this function converts numeric label indices and their names in text

def get_fashion_mnist_labels(labels): #@save
    text_labels = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5): #@save
    figsize = (num_cols*scale, num_rows*scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (ax, img) in enumerate(zip(axes, imgs)):
    
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
            
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        
        if titles:
            ax.set_title(titles[i])
    return axes 

"""
#Images with labels
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
show_images(X.reshape(18,28,28), 2, 9, titles=get_fashion_mnist_labels(y))
"""

#Reading a minitbatch
batch_size = 256

def get_dataloader_workers(): #@save
    """ Use 4 processors to read the data """
    return 4


"""
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
timer = d2l.Timer()

for X, y in train_iter:
    continue

print(f'{timer.stop():.2f} sec')
print("finished init")
"""

#Checking through both training and validation dataset
def load_data_fashion_mnist(batch_size, resize=None): #@save
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
        
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, transform=trans, download=True)
    
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
           data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))


train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
    
    
"""
Summary:
    -Fashion-MNIST is an apparel classification dataset consisting of images representing 10 categories. 
We will use this dataset in next sections and chapters to evaluate various classification algorithms.
    -We store the shape of any image with height h width w pixels as h*w or (h,w)
    -Data iterators are key component for efficient performance. Rely on well-implemented data iterators that exploit
high-performance computing to avoid slowing down your training loop.

"""


# In[ ]:





# In[ ]:




