
# coding: utf-8

# In[1]:


import utils.config as Config
import utils.model as model
from utils.pascal_voc import VOC
import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[2]:


train_path=Config.tfrecord_path+'train/'
val_path=Config.tfrecord_path+'val/'


# In[3]:


#读取tfrecord数据
def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer(filename)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.string),
                                               'img_raw' : tf.FixedLenFeature([], tf.string),
                                           })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [Config.image_size, Config.image_size, 3])
    img = (tf.cast(img, tf.float32)/255.0-0.5)*2
    label = tf.decode_raw(features['label'], tf.float64)
    label = tf.reshape(label, [Config.cell_size,Config.cell_size,25])
    return img,label


#读取训练集

# train_path=Config.tfrecord_path+'train/'
# train_files = [f for f in os.listdir(train_path) ]
# train_name=[train_path+fs for fs in train_files]
# train_img, train_label = read_and_decode([train_path+'10.tfrecords'])

#使用shuffle_batch可以随机打乱输入
train_img, train_label = read_and_decode([train_path+'train.tfrecords'])
train_img_batch, train_label_batch = tf.train.batch([train_img, train_label],
                                                batch_size=1, capacity=1000)


# In[4]:


sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)
for k in range(10):
    x1, y1 = sess.run([train_img_batch,train_label_batch])
    image=np.squeeze(x1)/2+0.5
    label=np.squeeze(y1)
    
    for i in range(7):
        for j in range(7):
            if label[i,j,0]==1:
                x=int(label[i,j,1])
                y=int(label[i,j,2])
                w=int(label[i,j,3]/2)
                h=int(label[i,j,4]/2)
                cv2.rectangle(image, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(image, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
    cv2.imshow('im',image)
    cv2.waitKey(0)
    cv2.destroyWindow('im')
cv2.waitKey(0)
cv2.destroyAllWindows()
sess.close()
coord.request_stop()


# In[5]:


#读取tfrecord数据
def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer(filename)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.string),
                                               'img_raw' : tf.FixedLenFeature([], tf.string),
                                           })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [Config.image_size, Config.image_size, 3])
    img = (tf.cast(img, tf.float32)/255.0-0.5)*2
    label = tf.decode_raw(features['label'], tf.float64)
    label = tf.reshape(label, [Config.cell_size,Config.cell_size,25])
    return img,label




#读取验证集
val_path=Config.tfrecord_path+'val/'
val_files = [f for f in os.listdir(val_path) ]
val_name=[val_path+fs for fs in val_files]
print(val_name)
val_img, val_label = read_and_decode([Config.tfrecord_path+'val/val.tfrecords'])

#使用shuffle_batch可以随机打乱输入
val_img_batch, val_label_batch = tf.train.batch([val_img, val_label],
                                                batch_size=1, capacity=1000)


# In[6]:


sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)
for k in range(10):
    x1, y1 = sess.run([val_img_batch,val_label_batch])
    image=np.squeeze(x1)
    label=np.squeeze(y1)
    
    for i in range(7):
        for j in range(7):
            if label[i,j,0]==1:
                x=int(label[i,j,1])
                y=int(label[i,j,2])
                w=int(label[i,j,3]/2)
                h=int(label[i,j,4]/2)
                cv2.rectangle(image, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(image, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
    cv2.imshow('im',image/2+0.5)
    cv2.waitKey(0)
    cv2.destroyWindow('im')
cv2.waitKey(0)
cv2.destroyAllWindows()
sess.close()
coord.request_stop()

