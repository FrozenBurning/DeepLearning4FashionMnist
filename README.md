<!--
 * @Description: 
 * @Author: Zhaoxi Chen
 * @Github: https://github.com/FrozenBurning
 * @Date: 2020-03-14 13:06:12
 * @LastEditors: Zhaoxi Chen
 * @LastEditTime: 2020-03-14 13:13:29
 -->
# Deep Learning On Fashion-Mnist

**2017011552 陈昭熹**

## 文件结构说明

### 文件夹

- report/ 报告
- images/ 相关图片

### 源码文件

#### 模型

- CNN4Module.py  四层全卷积网络模型
- myresnet18.py  Resnet18模型
- pretrain_res18.py  改进Resnet18模型
- inceptionNet.py InceptionNet模型

#### 训练

对应不同网络的训练脚本

- local_cnn4_valid.py 
- local_inception_valid.py
- local_resnet_valid.py

加载模型用于生成预测csv

- LoadModel.py

#### 一些功能

- divider.py  原始数据集分割
- grid_search.py  网格搜索法确定超参数
- visualize*  可视化