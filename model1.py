# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 23:06:23 2018

@author: apgt
"""

import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix

### Parámetros ###
beta = 1.0
learn_rate = 0.25
momentum = 0.9

### Funciones para entrenar la red neuronal ###
def sigmoid(x):
    return(1.0/(1.0+np.exp(-1*beta*x)))

def forward(input_layer,weight_layer):
    return(sigmoid(np.dot(input_layer,weight_layer)))

def update(w,t_pr,inp,t):
    n_data=len(w)
    for i in range(n_data):
        w[i]=w[i]-learn_rate*(t_pr-t)*inp[i]

def train2(input_data,w1,w2,ite,target):
    error = 0
    predic = np.zeros(len(target))
    updatew1, updatew2 = 0, 0
    #Forward
    for k in range(ite):
        if k%100==0:
            print(k)
        hid_n = forward(input_data,w1)
        hid_n = np.concatenate((hid_n,-np.ones((np.shape(input_data)[0],1))),axis=1)
        predic = forward(hid_n,w2).flatten()
        #Error
        error = 0.5*np.sum((predic-target)**2)
        deltao = beta*(predic-target)*predic*(1.0-predic)
        n = len(deltao)
        deltao = deltao.reshape(n,1)
        deltah = hid_n*beta*(1.0-hid_n)*(np.dot(deltao,np.transpose(w2)))
        #updatew1 = eta*(np.dot(np.transpose(inputs),deltah[:,:-1])) + self.momentum*updatew1
        #updatew2 = eta*(np.dot(np.transpose(self.hidden),deltao)) + self.momentum*updatew2
        updatew1 = learn_rate*(np.dot(np.transpose(input_data),deltah[:,:-1])) + momentum*updatew1
        updatew2 = learn_rate*(np.dot(np.transpose(hid_n),deltao)) + momentum*updatew2
        w1 -= updatew1
        w2 -= updatew2
    print('Error:', error)
    

def backwards(weight):
    pass

def recall(input_data,w1,w2):
    hid_n = forward(input_data,w1)
    hid_n = np.concatenate((hid_n,-np.ones((np.shape(input_data)[0],1))),axis=1)
    predic = forward(hid_n,w2).flatten()   
    return(predic)


data_in = np.array([[0,0],[0,1],[1,0],[1,1]])
out = np.array([0,1,1,0])
#inputs = np.concatenate((data_in,-np.ones((np.shape(data_in)[0],1))),axis=1)

"""
train2(inputs,weights1,weights2,500,out)
test = np.array(recall(inputs,weights1,weights2))
test = test.flatten()
test = np.where(test>0.5,1,0)
print(test)
print(confusion_matrix(out,test))
"""

### Archivo de entrada ###
data = np.genfromtxt('train_data_1.csv',delimiter=',',\
skip_header=1,dtype=float)

### Normalizar los datos ###
data[:,3] = (data[:,3] - data[:,3].mean(axis=0))/data[:,3].var(axis=0)

### Ordenar al azar los datos ###
random = list(range(np.shape(data)[0]))
np.random.shuffle(random)
data = data[random,:]

### Separar en training, validation y test 50:25:25 ###
training = data[::2,:]
validation = data[1::4,:]
test = data[3::4,:]

### Separar en input y output ###
train_in = training[:,1:]
train_out = training[:,0]
valid_in = validation[:,1:]
valid_out = validation[:,0]
test_in = test[:,1:]
test_out = test[:,0]

nin = np.shape(train_in)[1]
nout = 1
nhidden = 2
# Initialise network
weights1 = (np.random.rand(nin+1,nhidden)-0.5)*2/np.sqrt(nin)
weights2 = (np.random.rand(nhidden+1,nout)-0.5)*2/np.sqrt(nhidden)
inputs = np.concatenate((train_in,-np.ones((np.shape(train_in)[0],1))),axis=1)

train2(inputs,weights1,weights2,50,train_out)
test_in = np.concatenate((test_in,-np.ones((np.shape(test_in)[0],1))),axis=1)
test = np.array(recall(test_in,weights1,weights2))
test = test.flatten()
test = np.where(test>0.5,1,0)
#print(test)
#print(confusion_matrix(out,test))