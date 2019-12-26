#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'miyuan'
__mtime__ = '2019/7/10'
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
from sklearn.model_selection import train_test_split
import time

trainset = sklearn.datasets.load_files(container_path='data', encoding='UTF-8')
trainset.data, trainset.target = separate_dataset(trainset, 1.0)
# print(trainset.target_names)
# print(len(trainset.data))
# print(len(trainset.target))

# for i in range(len(trainset.target)):
#     if i<10:
#         print(trainset.target[i])
#         i+=1

ONEHOT = np.zeros((len(trainset.data), len(trainset.target_names)))
ONEHOT[np.arange(len(trainset.data)), trainset.target] = 1.0
# print(ONEHOT)
train_X, test_X, train_Y, test_Y, train_onehot, test_onehot = train_test_split(trainset.data, trainset.target, ONEHOT,
                                                                               test_size=0.2)
concat = ' '.join(trainset.data).split()
vocabulary_size = len(list(set(concat)))
data, count, dictionary, rev_dictionary = build_dataset(concat, vocabulary_size)
# print('vocab from size:%d'%(vocabulary_size))
# print('Most common words',count[4:10])
# print('sample data',data[:10],[rev_dictionary[i] for i in data[:10]])
GO = dictionary['GO']
PAD = dictionary['PAD']
EOS = dictionary['EOS']
UNK = dictionary['UNK']


class Model:
    def __init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate):
        def cells(reuse=False):
            # 基本可更改点1
            return tf.nn.rnn_cell.BasicRNNCell(size_layer, reuse=reuse)

        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.float32, [None, dimension_output])
        encoder_embeddings = tf.Variable(tf.random_uniform([dict_size, embedded_size], -1, -1))
        encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings, self.X)
        rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
        outputs, _ = tf.nn.dynamic_rnn(rnn_cells, encoder_embedded, dtype=tf.float32)
        W = tf.get_variable('w', shape=(size_layer, dimension_output))
        b = tf.get_variable('b', shape=(dimension_output), initializer=tf.zeros_initializer())
        self.logits = tf.matmul(outputs[:, -1], W) + b
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


size_layer = 128
num_layer = 2
embedded_size = 2
dimension_output = len(trainset.target_names)
learning_rate = 1e-3
maxlen = 500
batch_size = 128

tf.reset_default_graph()
sess = tf.InteractiveSession()
# sess=tf.InteractiveSession(config =tf.ConfigProto(log_device_placement =True))
model = Model(size_layer, num_layer, embedded_size, vocabulary_size + 4, dimension_output, learning_rate)
sess.run(tf.global_variables_initializer())

EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 5, 0, 0, 0
while True:
    lasttime = time.time()
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
        print('break epoch:%d\n' % (EPOCH))
        break
    train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
    for i in range(0, (len(train_X) // batch_size) * batch_size, batch_size):
        batch_x = str_idx(train_X[i:i + batch_size], dictionary, maxlen)
        acc, loss, _ = sess.run([model.accuracy, model.cost, model.optimizer],
                                feed_dict={model.X: batch_x, model.Y: train_onehot[i:i + batch_size]})
        train_acc += acc
        train_loss += loss

    for i in range(0, (len(test_X) // batch_size) * batch_size, batch_size):
        batch_x = str_idx(test_X[i:i + batch_size], dictionary, maxlen)
        acc, loss = sess.run([model.accuracy, model.cost],
                             feed_dict={model.X: batch_x, model.Y: test_onehot[i:i + batch_size]})
        test_acc += acc
        test_loss += loss

    train_loss /= (len(train_X) // batch_size)
    train_acc /= (len(train_X) // batch_size)
    test_loss /= (len(test_X) // batch_size)
    test_acc /= (len(test_X) // batch_size)

    if test_acc > CURRENT_ACC:
        print('epoch :%d,pass acc:%f,current_acc:%f' % (EPOCH, CURRENT_ACC, test_acc))
        CURRENT_ACC = test_acc
        CURRENT_CHECKPOINT = 0
    else:
        CURRENT_CHECKPOINT += 1

    print('time taken:', time.time() - lasttime)
    print('epoch :%d,training loss:%f,training acc:%f,valid_loss:%f,valid_acc:%f' % (
    EPOCH, train_loss, train_acc, test_loss, test_acc))

    EPOCH += 1

logits = sess.run(model.logits, feed_dict={model.X: str_idx(test_X, dictionary, maxlen)})
print(metrics.classification_report(test_Y, np.argmax(logits, 1), target_names=trainset.target_names))
