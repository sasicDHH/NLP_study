[NLP 计划]
---

[datawhale]
---
####一. 《[Task1 数据集探索 (2 days)](https://shimo.im/docs/pxzFefyYd7wLIcct)》
    
    1. Tensorflow安装GPU版
    实践：1.   Ubuntu1804 安装Ｎ卡GPU
    配置：
    NVIDIA Driver Version 418.67
    CUDA Version 10.0
    Cudnn Version 7.6.0
    Ubuntu 18.04
    Python Version 3.7.3
    Graphics card type GTX 1050Ti
    
    2. Tensorflow基础复习
    图graphs
    会话session
    张量tensor
    变量variable
    feed
    fetch
    
    
    
[IMDB]
---
    1.  IMDB下载
```python
import tensorflow as tf
from tensorflow import keras
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data('*****/imdb.npz',num_words=15000)
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels))) 
```
    2.  IMDB数据探索
   参考1.  [影评文本分类](https://tensorflow.google.cn/tutorials/keras/basic_text_classification)
   参考2.  [Kesci对IMDB电影评论进行情感分析 ](https://www.kesci.com/home/project/5b6c05409889570010ccce90)
   
[THUCNews]
---
    
    1. THUCNews下载
    THUCNews数据子集：https://pan.baidu.com/s/1hugrfRu 密码：qfud
    
    2. THUCNews数据探索
    
   参考1.  [CNN字符级中文文本分类-基于TensorFlow实现](https://blog.csdn.net/u011439796/article/details/77692621)
   参考2.  [text-classification-cnn-rnn](https://github.com/gaussic/text-classification-cnn-rnn/blob/master/data/cnews_loader.py)
    
[模型评估]
---
    1. 召回率 recall
    recall= TP/(TP+FN)
    
    2. 准确率  precison    查准率
    在所有预测积极标签中，真的积极标签占比
    precison=TP/(TP+FP)
    
    3. ROC曲线
    Receiver Operating Characteristic,    "接受者操作特性曲线"
        i.    分析
        ①    ROC曲线上的每一个点对应于一个threshold
        ②    Threshold最大时，TP=FP=0，对应于原点；Threshold最小时，TN=FN=0，对应于右上角的点(1,1)
        ii.    判断标准
        ①    曲线距离左上角越近，分类器效果越好（对角点线完全随机分类）
        iii.    优点
        ①    对类分布/不平衡数据集不敏感（比如癌症检测）
        
    4. AUC曲线
    area under the curve
        i.    产生原因
        ①    ROC无法数字量化，需要可视化判断
        ii.    判断标准
        AUC越大，分类器分类效果越好（完全随机分类为0.5）
        iii.    优点
        ①    量化指标
        ②    跟Threshold无关了
    
    5. PR曲线 
