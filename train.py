
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


# In[2]:


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

train_path=Config.tfrecord_path+'train/'
train_img, train_label = read_and_decode([train_path+'train.tfrecords'])

#使用shuffle_batch可以随机打乱输入
train_img_batch, train_label_batch = tf.train.shuffle_batch([train_img, train_label],
                                                batch_size=Config.batch_size, capacity=1000,
                                               min_after_dequeue=500)

#读取验证集
val_path=Config.tfrecord_path+'val/'
val_img, val_label = read_and_decode([val_path+'val.tfrecords'])

#使用shuffle_batch可以随机打乱输入
val_img_batch, val_label_batch = tf.train.shuffle_batch([val_img, val_label],
                                                batch_size=Config.batch_size, capacity=1000,
                                                min_after_dequeue=500)


# In[3]:


sess=tf.InteractiveSession()
model=model.Model()
data=VOC()
saver=tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='yolo'),max_to_keep=3)
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)
train_writer=tf.summary.FileWriter(Config.train_graph,sess.graph)
val_writer = tf.summary.FileWriter(Config.val_graph, sess.graph)
#载入下载预训练模型
if os.path.isfile(Config.small_path):
    print('重载预训练模型')
    saver.restore(sess,Config.small_path)
t1=time.time()
train_losses=[]
val_losses=[]
max_loss=1
for i in range(Config.epoch):
    
    total_loss=0
    total_val_loss=0
    for k in range(len(data.get_labels_train)//Config.batch_size+1):
        x1, y1 = sess.run([train_img_batch,train_label_batch])
        loss,train_summaries,global_step,_=sess.run([model.total_loss,model.merged_summary,model.global_step,model.train_op],feed_dict={
            model.images:x1,model.labels:y1})
        train_writer.add_summary(train_summaries, global_step)
        total_loss+=loss
        if (k+1) % Config.checkpont==0:
            for j in range(len(data.get_labels_val)//Config.batch_size+1):
                x2, y2 = sess.run([val_img_batch,val_label_batch])
                loss_val,val_summaries=sess.run([model.total_loss,model.merged_summary],feed_dict={
                    model.images:x2,model.labels:y2})
               
                total_val_loss+=loss_val
            val_writer.add_summary(val_summaries,global_step)
            if (total_val_loss/len(data.get_labels_val))<max_loss:
                max_loss=total_val_loss/len(data.get_labels_val)
                saver.save(sess=sess, save_path=Config.model_path+'model.ckpt',global_step=(model.global_step))
            print(('epoch:{}/{}').format(i,Config.epoch))
            print(('batch: {}, train loss: {} , val loss: {} ,time: {}').format(k,total_loss/(Config.checkpont*Config.batch_size),total_val_loss/len(data.get_labels_val),
                                                                               time.time()-t1))
            t1=time.time()
            train_losses.append(total_loss/(Config.checkpont*Config.batch_size))
            val_losses.append(total_val_loss/len(data.get_labels_val))
            total_loss=0
            total_val_loss=0
            
coord.request_stop()
coord.join(threads)            
sess.close()



# In[ ]:


plt.title('Result Analysis')
plt.plot(list(range(len(train_losses))), train_losses, color='green', label='training loss')
plt.plot(list(range(len(val_losses))), val_losses, color='red', label='val loss')

plt.legend() # 显示图例

plt.xlabel('iteration times')
plt.ylabel('rate')
plt.show()

