# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 23:58:58 2017

@author: ARH
"""

"""import images to a x variable and train the data"""

import tensorflow as tf 
import sys
import os 
import math as m
import argparse
import random as ran
from matplotlib import pyplot as plt
import numpy as np
#import mnist as input_data
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.examples.tutorials.mnist import mnist
#creating a local folder
mnist=input_data.read_data_sets("./data/", one_hot=True)

"""
with tf.Session() as sess:
    #access first image
    for i in range(100):
        first_image = mnist.train.images[0]
        first_image = np.array(first_image, dtype='uint8')
        pixels = first_image.reshape((28, 28))
        plt.imshow(pixels, cmap='gray')
        print(i)
"""     

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

def display_mult_flat(start, stop):
    images = x_train[start].reshape([1,784])
    for i in range(start+1,stop):
        images = np.concatenate((images, x_train[i].reshape([1,784])))
    plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    plt.show()
    
def display_compare(num):
    # THIS WILL LOAD ONE TRAINING EXAMPLE
    x_train = mnist.train.images[num,:].reshape(1,784)
    y_train = mnist.train.labels[num,:]
    # THIS GETS OUR LABEL AS A INTEGER
    label = y_train.argmax()
    # THIS GETS OUR PREDICTION AS A INTEGER
    prediction = sess.run(y, feed_dict={x: x_train}).argmax()
    plt.title('Prediction: %d Label: %d' % (prediction, label))
    plt.imshow(x_train.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
    plt.show()

    
#x_train,y_train=TRAIN_SIZE(55000)
#x_test,y_test=TEST_SIZE(1000)  
#learning_rate =0.1
#train_steps = 100 

"""weight adjustment """
def softmax(x=[],*args):
    ip=0
    for i in range(10):
        ip=ip+x[i]
    for i in range(10):
        x[i]=m.exp(x[i])/m.exp(ip)
    return x


def prd(to=[],*args):
    lrg=to[0]
    idx=0
    for i in range(1,len(to)):
        print(to[i])
        if(lrg<to[i]):
            idx=i
            lrg=to[i]
    return idx
  
def tr_mnist(lr,x=[],t=[],*args):
    z=tf.zeros([10])
    #x=tf.placeholder([None,728])
    W = tf.Variable(tf.zeros([784,10]))
    B = tf.Variable(tf.zeros([10]))
    init = tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)
    w=sess.run(W)
    b=sess.run(B)
    zw=sess.run(z)
    for i in range(784):
        for n in range(10):
            #WiXi calculation
            zw[n]=zw[n]+ w[i][n]*x[:,i]
            #y = WiXi + b
    for j in range(10):
         zw[j]=zw[j]+b[j]
    #activation function 
    to=softmax(zw)
    #idx=prd(to)
    #print(idx)
    #weight updation
    er=np.zeros([10])
    for k in range(10):
        er[k]=(t[0][k]-to[k])
    for j in range(784):
        for i in range(0,10):
            w[j][i]=lr*er[i]*x[:,j]
    idx=prd(er)
    for i in range(10):
        if(idx==i):
            print(" prediction :",i)
    
            
x = (mnist.train.images[:1,:])
y = (mnist.train.labels[:1,:])
             
tr_mnist(0.1,x,y)

#print(we)       
#for i in range(0):        
#sess = tf.Session()
#x=tf.placeholder(tf.float32,shape=[None,784])
#y_=tf.placeholder(tf.float32,shape=[None,10])
#weight and bias initailzing 

#activation of output from y = b + w*x
#y = tf.nn.softmax(tf.matmul(x,W) + b)
#the error calculation and adjust of weight and bias
#according to it's softmax activation and comparing
#the result with closeness to 0 or 1 
#if closer to 1 raise the value by .1 
#if closer to 0 the adds a large neg value
#so that the accuracy increace 
#eg:
#  not actual results [.001,.9,.2]
#applying corss entorpy we get
#[-.0046,.91,-.26]
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#init = tf.global_variables_initializer()
#sess.run(init)
#gradient descent is used to minimize the lose ie cross_entropy
#more accuracy
#training = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
"""
for i in range(train_steps+1):
    sess.run(training, feed_dict={x: x_train, y_: y_train})
    if i%1000 == 0:
        print('Training Step:' + str(i) + '  Accuracy =  ' + str(sess.run(accuracy, feed_dict={x: x_test, y_: y_test})) + '  Loss = ' + str(sess.run(cross_entropy, {x: x_train, y_: y_train})))
#W = W + 0.4*(tf.matmul)
    
    
for i in range(10):
    plt.subplot(2, 5, i+1)
    weight = sess.run(W)[:,i]
    plt.title(i)
    plt.imshow(weight.reshape([28,28]), cmap=plt.get_cmap('seismic'))
    frame1 = plt.gca()
frame1.axes.get_xaxis().set_visible(False)
frame1.axes.get_yaxis().set_visible(False)

plt.show()"""
"""
x_train = (mnist.train.images[:1,:])
#print(x_train)

ne=x_train.reshape([28,28])
plt.imshow(ne,cmap=plt.get_cmap('flag_r'))
plt.show()

for i in range(783):
    if(x_train[0][i] > 0):
        x_train[0][i]=1;
    
ne=x_train.reshape([28,28])
plt.imshow(ne,cmap=plt.get_cmap('flag_r'))
plt.show()

#print(x_train[1])"""
#print(ne)
#display_compare(ran.randint(0, 55000))




#image = x_train[98].reshape([28,28])
#plt.imshow(image, cmap=plt.get_cmap('gray_r'))
#plt.show()