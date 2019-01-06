# tensorflow-YOLO1
目标检测yolo算法，采用tensorflow框架编写，中文注释完全，含测试和训练，支持摄像头<br><br>

# 模型简介
## yolo v1
yolo1是端对端的目标检测模型，参考论文为[You Only Look Once:Unified, Real-Time Object Detection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf)<br>主要思想是将图片分割成cell_size * cell_size的格子，每个格子里只包含一个目标，通过网络来输出每个格子的目标值，其中判断格子中是否有目标即判断目标中心点是否在对应格子中。<br>
模型大致结构图如下：<br>
![](https://github.com/LeslieZhoa/-tensorflow-YOLO1-/blob/master/output/model.png)<br>
模型经过多层卷积和全连接层，将图片最终输出尺寸为[batch,cell_size * cell_size * (num_classes+ box_per_cell* 5)]。<br>简单介绍一下输出的表示:<br>
通过reshape分成[batch,cell_size,cell_size,num_classes]表示每个格子对应的类别;<br><br>
[batch_,cell_size,cell_size,box_per_cell]表示每个格子中是否存在目标的置信度，之所以选择两个box_per_cell是为了让预测精度更准确，通俗来讲就是三个臭皮匠顶一个诸葛亮;<br><br>
[batch,cell_size,cell_size,box_per_cell,4]表示每个格子每个选框中的目标框的坐标值，4个值分别为目标图片的中心横纵坐标偏移值x,y，目标图片的长宽w,h，但都经过一定归一化处理。x,y是相对于格子的偏移量经过了除以整个图片大小的归一化。<br><br>
举例说明:<br><br>
就是原图目标的中心坐标是x1,y1,原图宽高分别为w1,h1假设目标中心坐落在下标为[0,1]的格子里，即int(x1/image_size* cell_size),int(y1/image_size* cell_size)=0,1,此时对应格子的目标置信度应该都为1,x和y应该是相对于[0,1]这个格子的偏移量，具体算法是：x=x1/image_size* cell_size-0,y=y1/image_size* cell_size-1。<br><br>
w,h也进行归一化但还要开方，具体算法为：w=square(w1/image_size),h=square(h1/image_size),归一化可以把数值指定在一定范围有利于训练，把w,h开方，是因为w，h的值不同于x,y是格子的偏移量限定于一定区间，w,h是针对整个图片而言，变化没那么平缓，所以进行开方。<br><br>
真实训练数据也按上述方法来处理，只不过刚开始的shape是[cell_size,cell_size,4]然后将它reshape成[cell_size,cell_size,1,4]再复制成[batch,cell_size,cell_size,box_per_cell,4]<br><br>
关于损失函数计算有目标损失，无目标损失，类别损失，目标框损失，占比不同，实际显示图片要加上非极大值抑制，把两个很相近的目标框只保留置信度高的。<br>
## yolo v2
关于yolo v2 网上博客大致内容介绍很详细，可以参考论文[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)<br>
我主要介绍它的训练数据长什么样，这也是困扰我好久的。yolo v2 加了anchor box为上述每个格子提供多个目标的可能，真实值的目标要与anchor box计算iou值，大于阈值才保留，否则保留iou值最大的目标，这样label的shape就变成了[batch, cell_size, cell_size, N_ANCHORS, num_clsses+5],相关坐标x,y,w,h和yolo1处理方式也些许不同，感兴趣的同学可以去参考论文。<br>
# 代码介绍
代码只针对于yolo1的训练和测试
## 环境说明：
主要环境配置为：<br>
ubuntu 16.04<br>
python 3.5.5<br>
tensorflow 1.4.1<br>
opencv 3.4.1<br>
不知道windows可不可以，应该没问题
## 下载数据
训练数据下载[VOC](https://pan.baidu.com/s/10UHNDvhLA3-CwGOvA7TanQ)解压放置到data目录下，[预训练模型](https://pan.baidu.com/s/1RNzPt0naAT8AT-RTPrK1vw)放置到data目录下
## 代码介绍
data下放置训练数据和预训练模型和将数据生成的tfrecords文件<br>
graph保存训练过程中的训练集和验证集的graph<br>
model保存训练的最优model<br>
output是测试图片保存目录<br>
picture是测试图片放置目录<br>
utils包括配置文件config,模型文件model，数据处理文件psscal_voc<br>
image_test.py是判断生成tfrecords文件是否图片标注正确<br>
test.py是测试文件<br>
tfrecord.py是将数据处理成tfrecords格式<br>
train.py是训练文件
## 运行
首先可以手动修改config配置文件<br>
若要训练的话:<br>
运行python tfrecord.py 生成数据<br>
运行python train.py 训练数据<br><br>
若要测试:<br>
把自己喜欢图片放到picture内，本代码图片来源于百度图片<br>
查看[代码](https://github.com/LeslieZhoa/tensorflow-YOLO1/blob/master/utils/config.py#L60),确定你进行测试要使用的model，运行test.py<br>
本测试代码支持摄像头<br>
## 建议
建议下载预训练模型训练，训练次数不宜过长，否则过拟合很严重<br>
本代码只保存验证集上的最优模型<br>
代码参考[hizhangp](https://github.com/hizhangp/yolo_tensorflow)<br>
如有错误还请多多指正
# 结果展示
![](https://github.com/LeslieZhoa/-tensorflow-YOLO1-/blob/master/output/2007_000364.jpg)<br>
![](https://github.com/LeslieZhoa/-tensorflow-YOLO1-/blob/master/output/4.jpg)<br>
![](https://github.com/LeslieZhoa/-tensorflow-YOLO1-/blob/master/output/test.jpg)<br>
