# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 23:58:58 2017

@author: ARH
"""

"""import images to a x variable and train the data"""

import tensorflow as tf 
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
x_train,y_train=TRAIN_SIZE(2000)  
#Creating the model
train_steps=2000
lr=0.1
with tf.device("/gpu:0"):    
	x=tf.placeholder(tf.float32,shape=[None,784])
	y=tf.placeholder(tf.float32,shape=[None,10])
	to=tf.zeros([10])
	W = tf.Variable(tf.random_normal([784,10]))
	B = tf.Variable(np.random.rand(10))
	layer_2w=tf.Variable(tf.zeros([10,10]))
	layer_2b=tf.Variable(np.random.rand(10))
	layer_3w=tf.Variable(tf.zeros([10,10]))
	layer_3b=tf.Variable(np.random.rand(10))
init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
w=sess.run(W)
b=sess.run(B)
to=sess.run(to)
#layer2 Sessioni
l2w=sess.run(layer_2w)
l2b=sess.run(layer_2b)
l3w=sess.run(layer_3w)
l3b=sess.run(layer_3b)
#feeding data to x-image and y-label for training data
x=sess.run(x,feed_dict={x : x_train})
y=sess.run(y,feed_dict={y : y_train})
r=tf.zeros([10])
er=sess.run(r)

#training the neural_net
for xc in range(train_steps):
    z=tf.zeros([10])
    zw=sess.run(z)
    #layer one input weighted sum 
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
	
    for i in range(10):
        er[i]=(y[xc][i]-to[i])
		
    for j in range(784):
        for i in range(10):
            #weight & bias updation with error computed
            w[j][i]=w[j][i]+lr*er[i]*x[n][j]#new weight
            b[i]=b[i]+lr*er[i]#new bias
    
    
	#layer two 
    z2=tf.zeros([10])
    zw2=sess.run(z2)
    for i in range(10):
        for j in range(10):
			#weighted_input
            zw2[j]=zw2[j] + (l2w[i][j])*to[i]
            
			#layer two bias addition 
    for j in range(10):
        zw2[j]=zw2[j]+l2b[j]

    soft_l2=tf.nn.softmax(zw2)
    sess3=tf.Session()
    soft_l2=sess3.run(soft_l2)
	
	
    for k in range(10):
        er[k]=(y[xc][k]-soft_l2[k])
    for j in range(10):
        for i in range(10):
            #weight & bias updation with error computed
            l2w[j][i]=l2w[j][i]+lr*er[i]*to[j]#new weight
            l2b[i]=l2b[i]+lr*er[i]#new bias
	
	
	#layer three
    z3=tf.zeros([10])
    zw3=sess.run(z2)
    for i in range(10):
        for j in range(10):
            zw3[j]=zw3[j] + (l3w[i][j])*soft_l2[i]

	#layer three bias addition 
    for j in range(10):
        zw3[j]=zw3[j]+l3b[j]

    soft_l3=tf.nn.softmax(zw3)
    sess4=tf.Session()
    soft_l3=sess3.run(soft_l3)
	
    for k in range(10):
        er[k]=(y[xc][k]-soft_l3[k])
    for j in range(10):
        for i in range(10):
            #weight & bias updation with error computed
            l3w[j][i]=l3w[j][i]+lr*er[i]*soft_l2[j]#new weight
            l3b[i]=l3b[i]+lr*er[i]#new bias
    #test predictioin
    lrg=soft_l3[0]
    idx=0
    for i in range(10):
        if(lrg<soft_l3[i]):
            idx=i
            lrg=to[i]
    for i in range(10):
        if(y[xc][i]==1):
            idxt=i
	
	
    sess2.close()
    sess3.close()
    sess4.close()
    print('target =',idxt,end=' ')
    print('prediction =',idx)

plt.plot(w)
plt.show()