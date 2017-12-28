# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 14:31:33 2017

@author: shryu8902
"""
import pandas as pd
import os
import numpy as np
import tensorflow as tf
#%%
def MAPE(y_true,y_pred):
    mape = np.mean(abs((y_true-y_pred)/y_true))
    return mape
#%%
def RMSE(y_true,y_pred):
    rmse = np.sqrt(np.mean((y_true-y_pred)**2))
    return rmse

#%%
def weight_gen(shape):
    #weight and bias generator
    if len(shape)==4:
        n_in=shape[2]
        n_out=shape[3]
    elif len(shape)==2:
        n_in=shape[0]
        n_out=shape[1]
    w=tf.Variable(tf.random_normal(shape)/np.sqrt(n_in/2))
    b=tf.Variable(tf.random_uniform([n_out], 0, 0.1 / np.sqrt(n_in)))
    return w, b
# %%
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):

        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
#%%
def val_provider(x,y,val_size):
    rand_idx = np.random.choice(x.shape[0],val_size,replace=False)
    x_val = x[rand_idx,:]
    y_val = y[rand_idx,:]
    x_train = np.delete(x,rand_idx,axis=0)
    y_train = np.delete(y,rand_idx,axis=0)
    return x_val,y_val,x_train,y_train
#%%    
def batch_provider(x,y,num_data,batch_size):
    rand_idx = np.random.choice(num_data,batch_size)
    if len(x.shape)==2:
        x_batch=x[rand_idx,:]
    else :
        x_batch = x[rand_idx,:,:]
    y_batch = y[rand_idx,:]

    # print(y_batch.shape)
    return x_batch, y_batch
#%%
def to_csv_file(data_list,filename):               
    data_list_csv = pd.DataFrame(data_list)
    data_list_csv.to_csv(os.path.join(os.getcwd(),filename),index=False)
#%%    
def read_csv_file(filename):    
    filepath=os.path.join(os.getcwd(),filename)
    data = pd.read_csv(filepath,index_col=0,header=0)
    data = data.values # array 형식으로 바꿔줌
    return data
#%%
def data_pre(filename,CNN=True):
    data = pd.read_csv(filename,header=None)
    ncol=int(data.shape[1]-1)
    V_range=range(int(ncol/3))
    I_range=range(int(ncol/3),int(2*ncol/3))
    T_range=range(int(2*ncol/3),int(ncol))
    train_range=range(0,501)
    test_range=range(501,631)

    VIT = data.values # change string into array

    xx_train = np.array(VIT[train_range, 0:ncol])
    yy_train = np.array(VIT[train_range, ncol:(ncol+1)])

    xx_test = np.array(VIT[test_range, 0:ncol])
    yy_test = np.array(VIT[test_range, ncol:(ncol+1)])

    # normalization
    train_max_V = np.max(xx_train[:, V_range])
    train_min_V = np.min(xx_train[:, V_range])
    train_max_I = np.max(xx_train[:, I_range])
    train_min_I = np.min(xx_train[:, I_range])
    train_max_T = np.max(xx_train[:, T_range])
    train_min_T = np.min(xx_train[:, T_range])
    train_max_C = np.max(yy_train)
    train_min_C = np.min(yy_train)

    x_train_norm_V = (xx_train[:, V_range] - train_min_V) / (train_max_V - train_min_V)
    x_train_norm_I = (xx_train[:, I_range] - train_min_I) / (train_max_I - train_min_I)
    x_train_norm_T = (xx_train[:, T_range] - train_min_T) / (train_max_T - train_min_T)

    y_train_norm = (yy_train - train_min_C) / (train_max_C - train_min_C)
    y_train_norm = np.reshape(y_train_norm,[-1,1])

    x_test_norm_V = (xx_test[:, V_range] - train_min_V) / (train_max_V - train_min_V)
    x_test_norm_I = (xx_test[:, I_range] - train_min_I) / (train_max_I - train_min_I)
    x_test_norm_T = (xx_test[:, T_range] - train_min_T) / (train_max_T - train_min_T)

    y_test_norm = (yy_test - train_min_C) / (train_max_C - train_min_C)
    y_test_norm = np.reshape(y_test_norm,[-1,1])
    if CNN==True:
        x_train_norm = np.dstack((x_train_norm_V, x_train_norm_I, x_train_norm_T))  # intergration
        x_test_norm = np.dstack((x_test_norm_V, x_test_norm_I, x_test_norm_T)) # intergration
    else:
        x_train_norm = np.hstack((x_train_norm_V, x_train_norm_I, x_train_norm_T))  # intergration
        x_test_norm = np.hstack((x_test_norm_V, x_test_norm_I, x_test_norm_T)) # intergration

    print('training data', x_train_norm.shape, y_train_norm.shape)
    print('test data', x_test_norm.shape, y_test_norm.shape)
    

    return x_train_norm, y_train_norm, x_test_norm, y_test_norm, train_min_C, train_max_C
