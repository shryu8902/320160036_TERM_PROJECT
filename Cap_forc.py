# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 23:44:58 2017

Convolutional Neural Network for battery

@author: shryu8902
"""
#%%
#import os
#path=os.getcwd()
#new_path=path+'\\CNN'
#os.chdir(new_path)
#%%
import tensorflow as tf
import numpy as np
from functions import MAPE, RMSE, val_provider, batch_provider, weight_gen, lrelu, data_pre, to_csv_file
import time
import matplotlib.pyplot as plt
#%%
# Hyperparameters =================================================

start_time = time.clock()
n_train_sample = 501 # 167 x 3
learning_rate = 0.0001
training_epochs = 500
batch_size = 50
display_step = 10
#Dropout
dropout = 0.5 
filename='totalbatterydata_10_rev.csv'        
#%%
#Mode selection
#Select validation mode or test mode
val_mode = False
#Select model between;Linear, FNN, CNN
Model = 'CNN'
#Select # of tests
num_test=1
#%%
####################Linear model##############################################3
# If using linear model, do following blocks 
##Read Data
if Model == 'Linear':
    x_train, y_train, x_test, y_test, train_min_C, train_max_C =  data_pre(filename,CNN=False)
    np.random.seed(100)
    if (val_mode==True):
        x_val,y_val,x_train,y_train=val_provider(x_train,y_train,50)
    else: 
        x_val=x_test
        y_val=y_test
## Set Network input and output
    x= tf.placeholder(tf.float32,[None,x_train.shape[1]])
    x_ = x
    y = tf.placeholder(tf.float32,[None,1])
    keep_prob=tf.placeholder("float")
## Set Network Structure
    w1,b1 = weight_gen([x_train.shape[1],1])
    y_pred = tf.matmul(x_,w1)+b1
#%%
####################FNN Structure##############################################3
# If using FNN, do following blocks 
##Read Data
elif Model == 'FNN':   
    x_train, y_train, x_test, y_test, train_min_C, train_max_C =  data_pre(filename,CNN=False)
    np.random.seed(100)
    if (val_mode==True):
        x_val,y_val,x_train,y_train=val_provider(x_train,y_train,50)
    else: 
        x_val=x_test
        y_val=y_test
## Network input
    x= tf.placeholder(tf.float32,[None,x_train.shape[1]])
    x_ = x
    y = tf.placeholder(tf.float32,[None,1])
    keep_prob=tf.placeholder("float")
## Network structure
    w1,b1 = weight_gen([x_train.shape[1],40])
    w2,b2 = weight_gen([40,1])
    L1 = lrelu(tf.matmul(x_, w1) + b1)
    L1_drop=tf.nn.dropout(L1,keep_prob)
    y_pred = tf.matmul(L1,w2)+b2
#%%
#######################CNN Structure###############################################
#If using CNN, do following blocks
#CNN case
##Read DATA
else:         
    x_train, y_train, x_test, y_test, train_min_C, train_max_C =  data_pre(filename,CNN=True)
    np.random.seed(100)
    if (val_mode==True):
        x_val,y_val,x_train,y_train=val_provider(x_train,y_train,50)
    else: 
        x_val=x_test
        y_val=y_test
# Network input
    x = tf.placeholder(tf.float32, [None,x_train.shape[1],x_train.shape[2]]) # of data type is 3x130
    x_ = tf.reshape(x, shape=[-1,1,x_train.shape[1],x_train.shape[2]])
    y = tf.placeholder(tf.float32, [None, 1]) # label(capacity)
    keep_prob=tf.placeholder("float")

################ modification ########################################

    # weight & bias
    w1,b1 = weight_gen([1,2,3,9])
    w2,b2 = weight_gen([1,2,9,4])
    w3,b3=weight_gen([1,1,1,1])
    w4,b4=weight_gen([1,3,1,1])
    w5,b5=weight_gen([4*3,1])
    w6,b6=weight_gen([2,1])
    #set network structure
    conv1 = tf.add(tf.nn.conv2d(x_,w1,strides=[1,1,2,1],padding="SAME"),b1)
    act1 = lrelu(conv1)
    conv2 = tf.add(tf.nn.conv2d(act1,w2,strides=[1,1,2,1],padding="SAME"),b2)
    act2 = lrelu(conv2)
    act2_flat = tf.reshape(act2,[-1,4*3])
    #act2_drop=tf.nn.dropout(act2_flat,keep_prob)
    #act3=lrelu(tf.matmul(act2_flat,w5)+b5)
    y_pred=tf.matmul(act2_flat,w5)+b5
#%%
#common command for all model
cost = tf.reduce_mean((y_pred-y)**2)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.initialize_all_variables()

#%% start session
###############################Session#####################################
# Run following blocks regardless of CNN, and FNN
# Initializing the variables
with tf.Session() as sess:
    MAPEs=[]
    RMSEs=[]
       
    # Find good initialization
    for k in range(num_test):    
        sess.run(init)
        loss_train=sess.run(cost, feed_dict={x:x_train,y:y_train,keep_prob:1})
        while (loss_train > 2):
            sess.run(init)
            loss_train=sess.run(cost, feed_dict={x:x_train,y:y_train,keep_prob:1})

        # Training cycle
        loss=[]
        for epoch in range(training_epochs):
        # Loop over all batches
            for i in range(n_train_sample//batch_size):
                x_batch, y_batch = batch_provider(x_train, y_train, num_data=x_train.shape[0], batch_size=batch_size)
                _,loss_train = sess.run([optimizer, cost], feed_dict={x: x_batch, y: y_batch,keep_prob:dropout})
                
            # Display logs per epoch step
            if (epoch+1) % display_step == 0:
                loss_val = sess.run(cost, feed_dict={x: x_val, y: y_val,keep_prob:1})

                loss.append([loss_train,loss_val])
                print("Epoch:", '%d' % (epoch+1), "traincost=",loss_train)
                print("Epoch:", '%d' % (epoch+1), "testcost=",loss_val)

            y_pred_data = sess.run(y_pred, feed_dict={x: x_val, y: y_val,keep_prob:1})
    
        to_csv_file(loss,'loss%d.csv' %k)
        # denormalization해서 실제 capacity와 MAPE 비교하는 과정 필요
        y_pred_train = sess.run(y_pred, feed_dict={x:x_train, y: y_train,keep_prob:1})
        y_pred_train_denorm = y_pred_train*(train_max_C - train_min_C) + train_min_C
        y_train_denorm = y_train * (train_max_C - train_min_C) + train_min_C
        y_pred_denorm = y_pred_data * (train_max_C - train_min_C) + train_min_C
        y_test_denorm = y_val * (train_max_C - train_min_C) + train_min_C
        to_csv_file([y_pred_denorm.reshape([-1]), y_test_denorm.reshape([-1])],'prediction%d.csv' %k)        
        mape_val=MAPE(y_test_denorm,y_pred_denorm)
        rmse_val=RMSE(y_test_denorm,y_pred_denorm)
        mape_train=MAPE(y_train_denorm,y_pred_train_denorm)
        rmse_train=RMSE(y_train_denorm,y_pred_train_denorm)
        MAPEs.append([mape_train,mape_val])
        RMSEs.append([rmse_train,rmse_val])
        print('MAPE / train:', mape_train,', val: ',mape_val)
        print('RMSE / train:', rmse_train,', val: ',rmse_val)        
        print("Elapsed time is '%s' seconds" % (time.clock() - start_time))
        #plot prediction and real data
        plt.figure(1)
        axisx=range(y_pred_denorm.shape[0])
        plt.plot(axisx,y_pred_denorm,'r--',label='Prediction')
        plt.plot(axisx,y_test_denorm,'b',label='Real')
        plt.xlabel('Cycle Index')
        plt.ylabel('Capacity')
        plt.legend()
        plt.grid(True)
        plt.show()
    MAPEs.append(list(np.mean(MAPEs,axis=0)))
    RMSEs.append(list(np.mean(RMSEs,axis=0)))
    to_csv_file(np.array(MAPEs).T.tolist(),'MAPEs.csv')
    to_csv_file(np.array(RMSEs).T.tolist(),'RMSEs.csv')