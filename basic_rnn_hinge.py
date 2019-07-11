#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'miyuan'
__mtime__ = '2019/7/11'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
               ┏┓     ┏┓
             ┏┛ ┻━━━┛ ┻┓
             ┃    ☃       ┃
             ┃  ┳┛┗┳   ┃
             ┃   ┻       ┃
             ┗━┓   ┏━┛
              ┃   ┗━━━┓
              ┃  神兽保佑  ┣┓
              ┃   永无BUG！ ┏┛
               ┗┓┓┏━┳┓┏┛
               ┃┫┫  ┃┫┫
               ┗┻┛  ┗┻┛
"""
from utils import *
import tensorflow as tf
from sklearn.model_selection import  train_test_split
import time

trainset=sklearn.datasets.load_files(container_path='data',encoding='UTF-8')
trainset.data,trainset.target=separate_dataset(trainset,1.0)
# print(trainset.target_names)

ONEHOT=np.zeros((len(trainset.data),len(trainset.target_names)))
ONEHOT[np.arange(len(trainset.data)),trainset.target]=1.0
train_X,test_X,train_Y,test_Y,train_onehot,test_onehot=train_test_split(trainset.data,trainset.target,ONEHOT,test_size=0.2)
concat=' '.join(trainset.data).split()
vocabulary_size=len(list(set(concat)))
data,count,dictionary,rev_dictionary=build_dataset(concat,vocabulary_size)
print('vocab from size:%d'%(vocabulary_size))
print('Most common words:',count[4:10])
print('Sample data',data[:10],[rev_dictionary[i] for i in data[:10]])

GO=dictionary['GO']
PAD=dictionary['PAD']
EOS=dictionary['EOS']
UNK=dictionary['UNK']

class Model:
    def __init__(self,size_layer,num_layers,embedded_size,dict_size,dimension_output,learning_rate):
        def cells(return=False):
            return tf.nn.rnn_cell.BasicRNNCell(size_layer,reuse=reuse)

    self.X=tf.placeholder(tf.int32,[None,None])
    self.Y=tf.placeholder(tf.float32,[None,dimension_output])
    encoder_embeddings=tf.Variable(tf.random_uniform([dict_size,embedded_size],-1,-1))
    encoder_embeded=tf.nn.embedding_lookup(encoder_embeddings,self.X)
    run_cells=tf.nn.rnn_cell.MultiRNNCell([cells() for  _ in range(num_layers)])
    outputs,_=tf.nn.dynamic_rnn(run_cells,encoder_embeded,dtype=tf.float32)
    W=tf.get_variable('W',shape=(size_layer,dimension_output),initializer=tf.orthogonal_initializer())
    b=tf.get_variable('b',shape=(dimension_output),initializer=tf.zeros_initializer())
    