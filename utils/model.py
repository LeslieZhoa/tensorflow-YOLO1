
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
import utils.config as Config
slim=tf.contrib.slim


# In[1]:


class Model:
    def __init__(self,training=True):
        self.batch_size=Config.batch_size
        self.learning_rate=Config.learning_rate
        self.decay_rate=Config.decay_rate
        self.momentum=Config.momentum
        self.classes=Config.classes_name
        self.num_classes=Config.num_class
        self.image_size=Config.image_size
        self.cell_size=Config.cell_size
        self.box_per_cell=Config.box_per_cell
        self.output_size=self.cell_size*self.cell_size*(self.num_classes+self.box_per_cell*5)
        self.scale=1.0*self.image_size/self.cell_size
        self.boundary1=self.cell_size*self.cell_size*self.num_classes
        self.boundary2=self.boundary1+self.cell_size*self.cell_size*self.box_per_cell
        self.object_scale=Config.object_scale
        self.no_object_scale=Config.no_object_scale
        self.class_scale=Config.class_scale
        self.coord_scale=Config.coordinate_scale
        self.training=training
        
        #形如[[0,0],[1,1]...[6,6]]与偏移量相加组成中心点坐标shpe[7,7,2]
        self.offset=np.transpose(np.reshape(np.array([np.arange(self.cell_size)]*self.cell_size*self.box_per_cell),
                                           (self.box_per_cell,self.cell_size,self.cell_size)),(1,2,0))
        self.images=tf.placeholder(tf.float32,[None,self.image_size,self.image_size,3])#做了归一化/255-0.5
        
        self.output=self.build_network(self.images,num_outputs=self.output_size,
                                      alpha=Config.alpha_relu,training=training)
        if training:
            with tf.variable_scope('train'):
                self.global_step=tf.contrib.framework.get_or_create_global_step()
                self.labels=tf.placeholder(tf.float32,[None,self.cell_size,self.cell_size,5+self.num_classes])#label按cell划分好
                self.loss_layer(self.output,self.labels)
                self.total_loss=tf.losses.get_total_loss()
                tf.summary.scalar('loss',self.total_loss)
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=self.decay_rate, momentum=self.momentum)
                self.train_op = optimizer.minimize(self.total_loss, global_step=self.global_step)
                self.merged_summary = tf.summary.merge_all()
    def build_network(self,images,num_outputs,alpha,keep_prob=Config.dropout,training=True,scope='yolo'):
        if not self.training:
            keep_prob=1.0
        with tf.variable_scope(scope):
           
            with slim.arg_scope([slim.conv2d,slim.fully_connected],activation_fn=leaky_relu(alpha),
                                weights_initializer=slim.xavier_initializer(),
                               weights_regularizer=slim.l2_regularizer(0.0005)):
                net = tf.pad(images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name = 'pad_1')
                net = slim.conv2d(net, 64, 7, 2, padding = 'VALID', scope = 'conv_2')
                net = slim.max_pool2d(net, 2, padding = 'SAME', scope = 'pool_3')
                net = slim.conv2d(net, 192, 3, scope = 'conv_4')
                net = slim.max_pool2d(net, 2, padding = 'SAME', scope = 'pool_5')
                net = slim.conv2d(net, 128, 1, scope = 'conv_6')
                net = slim.conv2d(net, 256, 3, scope = 'conv_7')
                net = slim.conv2d(net, 256, 1, scope = 'conv_8')
                net = slim.conv2d(net, 512, 3, scope = 'conv_9')
                net = slim.max_pool2d(net, 2, padding = 'SAME', scope = 'pool_10')
                net = slim.conv2d(net, 256, 1, scope = 'conv_11')
                net = slim.conv2d(net, 512, 3, scope = 'conv_12')
                net = slim.conv2d(net, 256, 1, scope = 'conv_13')
                net = slim.conv2d(net, 512, 3, scope = 'conv_14')
                net = slim.conv2d(net, 256, 1, scope = 'conv_15')
                net = slim.conv2d(net, 512, 3, scope = 'conv_16')
                net = slim.conv2d(net, 256, 1, scope = 'conv_17')
                net = slim.conv2d(net, 512, 3, scope = 'conv_18')
                net = slim.conv2d(net, 512, 1, scope = 'conv_19')
                net = slim.conv2d(net, 1024, 3, scope = 'conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope = 'pool_21')
                net = slim.conv2d(net, 512, 1, scope = 'conv_22')
                net = slim.conv2d(net, 1024, 3, scope = 'conv_23')
                net = slim.conv2d(net, 512, 1, scope = 'conv_24')
                net = slim.conv2d(net, 1024, 3, scope = 'conv_25')
                net = slim.conv2d(net, 1024, 3, scope = 'conv_26')
                net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name = 'pad_27')
                net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope = 'conv_28')
                net = slim.conv2d(net, 1024, 3, scope = 'conv_29')
                net = slim.conv2d(net, 1024, 3, scope = 'conv_30')
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                net = slim.flatten(net, scope = 'flat_32')
                net = slim.fully_connected(net, 512, scope = 'fc_33')
                net = slim.fully_connected(net, 4096, scope = 'fc_34')
                net = slim.dropout(net, keep_prob = keep_prob, is_training = training, scope = 'dropout_35')
                #最终输出层维度为[bacth,(7*7*(20+5*2))]
                net = slim.fully_connected(net, num_outputs, activation_fn = None, scope = 'fc_36')
        return net
    def cal_iou(self,boxes1,boxes2,scope='iou'):
        with tf.variable_scope(scope):
            boxes1=tf.stack([boxes1[:,:,:,:,0]-boxes1[:,:,:,:,2]/2.0,
                           boxes1[:,:,:,:,1]-boxes1[:,:,:,:,3]/2.0,
                           boxes1[:,:,:,:,0]+boxes1[:,:,:,:,2]/2.0,
                           boxes1[:,:,:,:,1]+boxes1[:,:,:,:,3]/2.0],axis=-1)
            boxes2=tf.stack([boxes2[:,:,:,:,0]-boxes2[:,:,:,:,2]/2.0,
                           boxes2[:,:,:,:,1]-boxes2[:,:,:,:,3]/2.0,
                           boxes2[:,:,:,:,0]+boxes2[:,:,:,:,2]/2.0,
                           boxes2[:,:,:,:,1]+boxes2[:,:,:,:,3]/2.0],axis=-1)
            #重叠区域左上角坐标
            a=tf.maximum(boxes1[:,:,:,:,:2],boxes2[:,:,:,:,:2])
            #重叠区域右下角坐标
            b=tf.minimum(boxes1[:,:,:,:,2:],boxes2[:,:,:,:,2:])
            #坐标之差，便于求取重叠部分面积
            c=tf.maximum(0.0,b-a)
            #重叠区域面积
            Overlap=c[:,:,:,:,0]*c[:,:,:,:,1]
            #总面积
            Area=tf.maximum((boxes1[:,:,:,:,2]-boxes1[:,:,:,:,0])*(boxes1[:,:,:,:,3]-boxes1[:,:,:,:,1])+                  (boxes2[:,:,:,:,2]-boxes2[:,:,:,:,0])*(boxes2[:,:,:,:,3]-boxes2[:,:,:,:,1])-Overlap,1e-10)
            iou=tf.clip_by_value(Overlap/Area, 0.0, 1.0)
        return iou
    def loss_layer(self,predicts,labels,scope='loss'):
        ''' predicts的shape是[batch,7*7*(20+5*2)]
            labels的shape是[batch,7,7,(5+20)]
            '''
        with tf.variable_scope(scope):
            #预测种类，boxes置信度，boxes坐标[x_center,y_center,w,h],坐标都除以image_size归一化,中心点坐标为偏移量，
            #w,h归一化后又开方，目的是使变化更平缓
            predict_classes=tf.reshape(predicts[:,:self.boundary1],
                                      [self.batch_size,self.cell_size,self.cell_size,self.num_classes])
            predict_scales=tf.reshape(predicts[:,self.boundary1:self.boundary2],
                                     [self.batch_size,self.cell_size,self.cell_size,self.box_per_cell])
            predict_boxes=tf.reshape(predicts[:,self.boundary2:],
                                    [self.batch_size,self.cell_size,self.cell_size,self.box_per_cell,4])
            #是否有目标的置信度
            response=tf.reshape(labels[:,:,:,0],
                               [self.batch_size,self.cell_size,self.cell_size,1])
            #boxes坐标处理变成[batch,7,7,2,4],两个box最终只选一个最高的，为了使预测更准确
            boxes=tf.reshape(labels[:,:,:,1:5],
                            [self.batch_size,self.cell_size,self.cell_size,1,4])
            boxes=tf.tile(boxes,[1,1,1,self.box_per_cell,1])/self.image_size
            classes=labels[:,:,:,5:]
            #offset形如[[[0,0],[1,1]...[6,6]],[[0,0]...[6,6]]...]与偏移量x相加
            #offset转置形如[[0,0,[0,0]...],[[1,1],[1,1]...],[[6,6]...]]与偏移量y相加
            #组成中心点坐标shpe[batch,7,7,2]是归一化后的值
            offset=tf.constant(self.offset,dtype=tf.float32)
            offset=tf.reshape(offset,[1,self.cell_size,self.cell_size,self.box_per_cell])
            offset=tf.tile(offset,[self.batch_size,1,1,1])
            
            predict_boxes_tran=tf.stack([(predict_boxes[:,:,:,:,0]+offset)/self.cell_size,
                                       (predict_boxes[:,:,:,:,1]+tf.transpose(offset,(0,2,1,3)))/self.cell_size,
                                        tf.square(predict_boxes[:,:,:,:,2]),
                                         tf.square(predict_boxes[:,:,:,:,3])],axis=-1)
            #iou的shape是[batch,7,7,2]
            iou_predict_truth=self.cal_iou(predict_boxes_tran,boxes)
            #两个预选框中iou最大的
            object_mask=tf.reduce_max(iou_predict_truth,3,keep_dims=True)
            #真实图中有预选框，并且值在两个预选框中最大的遮罩
            object_mask=tf.cast((iou_predict_truth>=object_mask),tf.float32)*response
            #无预选框遮罩
            noobject_mask=tf.ones_like(object_mask,dtype=tf.float32)-object_mask
            #真实boxes的偏移量
            boxes_tran=tf.stack([boxes[:,:,:,:,0]*self.cell_size-offset,
                                boxes[:,:,:,:,1]*self.cell_size-tf.transpose(offset,(0,2,1,3)),
                                tf.sqrt(boxes[:,:,:,:,2]),
                                tf.sqrt(boxes[:,:,:,:,3])],axis=-1)
            #分类损失
            class_delta=response*(predict_classes-classes)
            class_loss=tf.reduce_mean(tf.reduce_sum(tf.square(class_delta),axis=[1,2,3]),name='clss_loss')*self.class_scale
            #有目标损失
            object_delta=object_mask*(predict_scales-iou_predict_truth)
            object_loss=tf.reduce_mean(tf.reduce_sum(tf.square(object_delta),axis=[1,2,3]),name='object_loss')*self.object_scale
            #无目标损失
            noobject_delta=noobject_mask*predict_scales
            noobject_loss=tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta),axis=[1,2,3]),name='noobject_loss')*self.no_object_scale
            #选框损失
            coord_mask=tf.expand_dims(object_mask,4)
            boxes_delta=coord_mask*(predict_boxes-boxes_tran)
            coord_loss=tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta),axis=[1,2,3,4]),name='coord_loss')*self.coord_scale
            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)
            


# In[3]:


def leaky_relu(alpha):
    
    def op(inputs):
        return tf.nn.leaky_relu(inputs,alpha=alpha)
    return op

