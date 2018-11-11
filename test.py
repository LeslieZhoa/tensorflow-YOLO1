
# coding: utf-8

# In[1]:


import utils.config as Config
import utils.model as model
import os
import time
import cv2
import tensorflow as tf
import numpy as np


# In[2]:


sess=tf.InteractiveSession()
model=model.Model(training=False)
sess.run(tf.global_variables_initializer())
saver=tf.train.Saver(tf.global_variables())
boundary1=Config.cell_size*Config.cell_size*Config.num_class
boundary2=boundary1+Config.cell_size*Config.cell_size*Config.box_per_cell


# In[3]:




def main():
    if Config.model_type=='1':
        saver.restore(sess,Config.small_path)
        print('重载预训练模型')
    elif Config.model_type=='2':
        model_file=tf.train.latest_checkpoint(Config.model_path)
        saver.restore(sess,model_file)
        print('重载自己训练模型')
    else:
        print('滚去跑模型')
        return

    if Config.output_type=='1':
        for filename in os.listdir(Config.picture):
            image=cv2.imread(Config.picture+filename)
            read_image(image,filename)
        cv2.destroyAllWindows()
    if Config.output_type=='2':
        cap=cv2.VideoCapture(0)
        cap.set(3,Config.image_size)
        cap.set(4,Config.image_size)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(Config.output_path+'out.avi' ,fourcc,10,(640,480))
        while True:
            ret,frame = cap.read()
            if ret == True:
                result=detect(frame)
                draw_result(frame,result)
                a = out.write(frame)
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()


# In[4]:


def read_image(image,filename):
    result = detect(image)
    draw_result(image, result)
    cv2.imshow('im',image)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:        
        cv2.imwrite(Config.output_path + filename,image)


# In[5]:


#将边框加入图片
def draw_result(img,result):
    for i in range(len(result)):
        x=int(result[i][1])
        y=int(result[i][2])
        w=int(result[i][3]/2)
        h=int(result[i][4]/2)
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
        cv2.putText(img, result[i][0] + ' : %.2f' % result[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


# In[6]:


def detect(img):
    img_h,img_w,_=img.shape
    inputs=cv2.resize(img,(Config.image_size,Config.image_size)).astype(np.float32)
    inputs=(inputs/255.0-0.5)*2
    inputs=np.reshape(inputs,(1,Config.image_size,Config.image_size,3))
    result=detect_from_cvmat(inputs)[0]
    for i in range(len(result)):
        result[i][1] *= (1.0 * img_w / Config.image_size)
        result[i][2] *= (1.0 * img_h / Config.image_size)
        result[i][3] *= (1.0 * img_w / Config.image_size)
        result[i][4] *= (1.0 * img_h / Config.image_size)
    return result


# In[7]:


#将模型输出存入results
def detect_from_cvmat(inputs):
    output=sess.run(model.output,feed_dict={model.images:inputs})
    results=[]
    for i in range(output.shape[0]):
        results.append(interpret_output(output[i]))
    return results


# In[8]:


def interpret_output(output):
    #有目标情况下的分类概率
    probs=np.zeros((Config.cell_size,Config.cell_size,Config.box_per_cell,len(Config.classes_name)))
    #分类概率
    class_probs=np.reshape(output[0:boundary1],(Config.cell_size,Config.cell_size,Config.num_class))
    #boxes有目标概率
    scales=np.reshape(output[boundary1:boundary2],(Config.cell_size,Config.cell_size,Config.box_per_cell))
    #boxes及偏移量修正
    boxes=np.reshape(output[boundary2:],(Config.cell_size,Config.cell_size,Config.box_per_cell,4))
    offset=np.transpose(np.reshape(np.array([np.arange(Config.cell_size)]*
                                           Config.cell_size*Config.box_per_cell),
                                  [Config.box_per_cell,Config.cell_size,Config.cell_size]),(1,2,0))
    
    boxes[:,:,:,0]+=offset
    boxes[:,:,:,1]+=np.transpose(offset,(1,0,2))
    boxes[:,:,:,:2]=1.0*boxes[:,:,:,:2]/Config.cell_size
    boxes[:,:,:,2:]=np.square(boxes[:,:,:,2:])
    boxes*=Config.image_size
    
    for i in range(Config.box_per_cell):
        for j in range(Config.num_class):
            probs[:,:,i,j]=np.multiply(class_probs[:,:,j],scales[:,:,i])
    #保留分类概率大于阈值的 shape=[7,7,2,20]      
    filter_mat_probs=np.array(probs>=Config.threshold,dtype='bool')
    #保留的下标,长度为4的元组对应ceil,ceil,box_per_cell,class_prob，每个元组长度为保留boxes的个数
    filter_mat_boxes=np.nonzero(filter_mat_probs)
    #保留boxes的坐标
    boxes_filtered=boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
    #保留boxes的置信度
    probs_filtered=probs[filter_mat_probs]
    #选框目标类别编号
    classes_num_filtered=np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
    #选框概率从大到小的下标
    argsort=np.array(np.argsort(probs_filtered))[::-1]
    #概率由大到小boxes坐标
    boxes_filtered=boxes_filtered[argsort]
    #概率由大到小boxes置信度
    probs_filtered=probs_filtered[argsort]
    #概率由大到小类别
    classes_num_filtered=classes_num_filtered[argsort]
    
    #非极大值抑制，将iou大的删掉
    for i in range(len(boxes_filtered)):
        if probs_filtered[i]==0:
            continue
        for j in range(i+1,len(boxes_filtered)):
            if iou(boxes_filtered[i],boxes_filtered[j])>Config.IOU_threshold:
                probs_filtered[j]=0.0
    filter_iou=np.array(probs_filtered>0.0,dtype='bool')
    #非极大值抑制后的参数
    boxes_filtered=boxes_filtered[filter_iou]
    probs_filtered=probs_filtered[filter_iou]
    classes_num_filtered=classes_num_filtered[filter_iou]
    result=[]
    for i in range(len(boxes_filtered)):
        result.append([Config.classes_name[classes_num_filtered[i]],boxes_filtered[i][0],
                       boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])
    return result


# In[9]:


def iou(box1,box2):
    #重叠部分长高
    tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    if tb < 0 or lr < 0:
        intersection = 0
    else:
        intersection = tb * lr
    return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)


# In[10]:


if __name__=='__main__':
    main()

