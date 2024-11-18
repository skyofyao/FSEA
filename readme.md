##FSEA训练过程 

1. 把数据集中的图像与xml文件分别放到VOCdevkit/VOC2007相对应的JPEGImages文件夹与Annotations里面
VOC2007的ImageSets文件夹没有必要放置任何txt文件。

2.打开voc_annotation.py文件
调整训练集与测试集的比例大小（一般为8：2）
trainval_percent    = 0.8
train_percent       = 0.8

当主文件夹中出现了2007_train.txt与2007_val.txt后证明训练集与测试集已经调整完成。
此时自动在VOCdevkit/VOC2007/ImageSets文件夹中填写了训练集与测试集文件路径。

3.把model_data/voc_classes.txt改成（杂草）数据对应的的类别名称，顺序无所谓（改完之后，整个训练过程，不要再动，如果在调参阶段增加种类名称，直接在后边添加，不要改变前面的顺序）

4.运行train.py文件可以训练

基类训练阶段：
model_path = 'model_data/voc_weights_resnet.pth'  #加载预训练的权重文件

input_shape=[1024, 1024] #所有的图像大小都会被resize为1024*1024
pretrained =False  #如果设置了model_path权重加载，这里的pretrained没有任何意义。

Freeze_Epoch在基类训练时没有任何的意义，设置为0.
使用UnFreeze_Epoch 表示训练代数
训练中采用了学习率衰减，最小学习率为最大的0.01倍
optimizer_type  使用到的优化器种类，可选的有adam、sgd
                         当使用Adam优化器时建议设置  Init_lr=1e-4
                         当使用SGD优化器时建议设置   Init_lr=5e-3（我们使用的是SGD的优化器）
save_period  表示多少个epoch保存一次权重

新类训练阶段：
新的类别训练时要使用基类训练的权重替换model_data文件夹中的权重文件
将model_data/voc_classes.txt增加新的植株类别
Freeze_Epoch在新类训练时，首先设置为5.
UnFreeze_Epoch设置为20
后重新开始训练
Freeze_Epoch=0，UnFreeze_Epoch=20.
在训练中，权重会逐渐开放。

