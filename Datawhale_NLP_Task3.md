[任务] 
---
[打卡连接](https://shimo.im/sheets/SiiVAuSS5fcDos1X/MODOC)

    1.  TF-IDF原理。
    2.  文本矩阵化，使用词袋模型，以TF-IDF特征值为权重。（可以使用Python中TfidfTransformer库）
    3.  互信息的原理。
    4.  使用第二步生成的特征矩阵，利用互信息进行特征筛选。

[参考]
---
   1. [文本挖掘预处理之TF-IDF：文本挖掘预处理之TF-IDF - 刘建平Pinard - 博客园](https://www.cnblogs.com/pinard/p/6693230.html)
   2. [使用不同的方法计算TF-IDF值：使用不同的方法计算TF-IDF值 - 简书](https://www.jianshu.com/p/f3b92124cd2b)
   3. [sklearn-点互信息和互信息：sklearn：点互信息和互信息 - 专注计算机体系结构 - CSDN博客](https://blog.csdn.net/u013710265/article/details/72848755)
   4. [如何进行特征选择（理论篇）机器学习你会遇到的“坑”：如何进行特征选择（理论篇）机器学习你会遇到的“坑” ](https://baijiahao.baidu.com/s?id=1604074325918456186&wfr=spider&for=pc)
   
 
[TF-IDF]
---
TF-IDF(Term Frequency-Inverse DocumentFrequency, 词频-逆文件频率)

1.    作用：用于资讯检索和资讯探勘的常用加权技术
统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度
字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降
TF-IDF加权的各种形式常被搜寻引擎应用，作为文件与用户查询之间相关程度的度量或评级。
2.    原理
词频TF（item frequency）：某一给定词语在该文本中出现次数
公式：TF=词在文章出现次数/文章总词数

逆向文件频率IDF（inverse document frequency）：一个词语普遍重要性的度量
公式：DF=log(语料库的文档总数/(包含该词的文档数+1))
+1,避免分母为0
主要思想是：如果包含词条t的文档越少, IDF越大，则说明词条具有很好的类别区分能力。某一特定词语的IDF，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取对数得到。

TF-IDF=TF×IDF
结论：TF-IDF与一个词在文档中出现的次数成正比，与该词在整个语言中该出现的次数成反比

例1.
假如一篇文件的总词语数是100个，而词语“母牛”出现了3次，那么“母牛”一词在该文件中的词频就是3/100=0.03。一个计算文件频率 (DF) 的方法是测定有多少份文件出现过“母牛”一词，然后除以文件集里包含的文件总数。所以，如果“母牛”一词在1,000份文件出现过，而文件总数是10,000,000份的话，其逆向文件频率就是 log(10,000,000 / 1,000)=4。最后的TF-IDF的分数为0.03 *4=0.12。

[文本矩阵化]
---
采用sklearn
```python
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  

corpus=["I come to China to travel", 
    "This is a car polupar in China",          
    "I love tea and Apple ",   
    "The work is to write some papers in science"] 

vectorizer=CountVectorizer()

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))  
print tfidf
```

[互信息原理]
---
两个随机变量的互信息（Mutual Information，简称MI）或转移信息（transinformation）是变量间相互依赖性的量度
互信息(Mutual Information)是度量两个事件集合之间的相关性(mutual dependence)。
互信息是点间互信息（PMI）的期望值。
互信息最常用的单位是bit。
其衡量的是两个随机变量之间的相关性，即一个随机变量中包含的关于另一个随机变量的信息量。所谓的随机变量，即随机试验结果的量的表示，可以简单理解为按照一个概率分布进行取值的变量，比如随机抽查的一个人的身高就是一个随机变量。

点互信息PMI
点互信息PMI(Pointwise Mutual Information)这个指标来衡量两个事物之间的相关性
```python
    from sklearn import metrics as mr
    mr.mutual_info_score(label,x)
```

[互信息应用]
```python
from sklearn import metrics as mr

# 互信息(Mutual Information)
labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]
MI = mr.adjusted_mutual_info_score(labels_true, labels_pred)  
print("Matual Information = "+str(MI))
# Matual Information = 0.2250422831983088
```
