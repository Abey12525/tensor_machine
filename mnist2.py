# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:20:01 2017

@author: ARH
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 23:58:58 2017

@author: ARH
"""

"""import images to a x variable and train the data"""

import tensorflow as tf 
import math as m
from matplotlib import pyplot as plt
import numpy as np
#import mnist as input_data
from tensorflow.examples.tutorials.mnist import input_data

#from tensorflow.examples.tutorials.mnist import mnist
#creating a local folder
mnist=input_data.read_data_sets("./data/", one_hot=True)

def TRAIN_SIZE(num):
    print ('Total Training Images in Dataset = ' + str(mnist.train.images.shape))
    print ('--------------------------------------------------')
    x_train = mnist.train.images[:num,:]
    print ('x_train Examples Loaded = ' + str(x_train.shape))
    y_train = mnist.train.labels[:num,:]
    print ('y_train Examples Loaded = ' + str(y_train.shape))
    print('')
    return x_train, y_train

def TEST_SIZE(num):
    print ('Total Test Examples in Dataset = ' + str(mnist.test.images.shape))
    print ('--------------------------------------------------')
    x_test = mnist.test.images[:num,:]
    print ('x_test Examples Loaded = ' + str(x_test.shape))
    y_test = mnist.test.labels[:num,:]
    print ('y_test Examples Loaded = ' + str(y_test.shape))
    return x_test, y_test
"""def nextbatch(x,index):
    for i in range(100):"""

def display_digit(num):
    print(y_train[num])
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([28,28])
    plt.title('Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()
#loading data to x_train(image set) and y_train(label set)
x_train,y_train=TRAIN_SIZE(100)  
#Creating the model
train_steps=100
lr=0.1    
x=tf.placeholder(tf.float32,shape=[None,784])
y=tf.placeholder(tf.float32,shape=[None,10])
to=tf.zeros([10])
W = tf.Variable(tf.zeros([784,10]))
B = tf.Variable(tf.zeros([10]))
init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
w=sess.run(W)
b=sess.run(B)
to=sess.run(to)
#feeding data to x-image and y-label for training data
x=sess.run(x,feed_dict={x : x_train})
y=sess.run(y,feed_dict={y : y_train})
#image=x[998].reshape([28,28])
#plt.imshow(image, cmap=plt.get_cmap('gray_r'))
#plt.show()
r=tf.zeros([10])
er=sess.run(r)

#training the neural_net
for xc in range(train_steps):
    z=tf.zeros([10])
    zw=sess.run(z)
    for i in range(784):
        for n in range(10):
            #WiXi calculation
            zw[n]=zw[n] + (w[i][n])*(x[xc][i])
    #y = WiXi + b
    for j in range(10):
         zw[j]=zw[j]+b[j]
         
    #activation function 
    to=tf.nn.softmax(zw)
    sess2=tf.Session()
    to=sess2.run(to)
    #print(to[0])
    #error calculation for labels from input and desired 
    for k in range(10):
        er[k]=(y[xc][k]-to[k])
    for j in range(784):
        for i in range(0,10):
            #weight & bias updation with error computed
            w[j][i]=lr*er[i]*x[n][j]#new weight
            b[i]=lr*er[i]#new bias
    lrg=to[0]
    idx=0
    for i in range(10):
        if(lrg<to[i]):
            idx=i
            lrg=to[i]
    for i in range(10):
        if(y[xc][i]==1):
            idxt=i
    sess2.close()
    print('target =',idxt,end=' ')
    print('prediction =',idx)