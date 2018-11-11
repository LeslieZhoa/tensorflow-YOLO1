
# coding: utf-8

# In[1]:


import os
import tensorflow as tf 
import numpy as np
from utils.pascal_voc import VOC
import utils.config as Config


# In[2]:


train_path=Config.tfrecord_path+'train/'
val_path=Config.tfrecord_path+'val/'
data=VOC()


# In[4]:


writer = tf.python_io.TFRecordWriter(train_path+"train.tfrecords")
for k in range(len(data.get_labels_train)):


    imname=data.get_labels_train[k]['imname']
    flipped=data.get_labels_train[k]['flipped']
    image=data.read_image(imname,flipped)
    label=data.get_labels_train[k]['label']   

    label_raw=label.tobytes()
    img_raw = image.tobytes()              #将图片转化为原生bytes
    example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
    writer.write(example.SerializeToString())  #序列化为字符串
    print (k)

writer.close()


# In[3]:


#验证集tfrecord
writer = tf.python_io.TFRecordWriter(val_path+"val.tfrecords")
for k in range(len(data.get_labels_val)):


    imname=data.get_labels_val[k]['imname']
    flipped=data.get_labels_val[k]['flipped']
    image=data.read_image(imname,flipped)
    label=data.get_labels_val[k]['label']   

    label_raw=label.tobytes()
    img_raw = image.tobytes()              #将图片转化为原生bytes
    example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
    writer.write(example.SerializeToString())  #序列化为字符串
    print (k)

writer.close()

