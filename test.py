#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'miyuan'
__mtime__ = '2019/7/17'
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
import tensorflow as tf
test = [[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]]
with tf.Session() as sess:
    print(sess.run(tf.argmax(test,1)))