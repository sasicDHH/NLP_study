[任务]
---

    词袋模型：离散、高维、稀疏。
    分布式表示：连续、低维、稠密。word2vec词向量原理并实践，用来表示文本。
    word2vec1 word2vec 
    word2vec 中的数学原理详解（一）目录和前言 - peghoty - CSDN博客  https://blog.csdn.net/itplus/article/details/37969519
    word2vec原理推导与代码分析-码农场  http://www.hankcs.com/nlp/word2vec.html
    
[词袋模型]
---
    
    Bag-of-words model (BoW model) 
    原理：句子看着若干个单词的集合，不会考虑单词的出现顺序，仅仅考虑单词出现没有或者出现的频率
    通过n-gram来表征单词间的关联也会造成高维、稀疏的情况发生
    缺点：语义丢失， 表现为，词的顺序信息丢失，近义词没办法体现，假定词都是独立的，
    常见方法:one-hot,词频，tf-idf 

[分布式表示]
---
    
    distributional representation
    基本思想是把研究的对象表示成一个低维的稠密的实质的向量，那么这种向量的物理意义就是在于它能够把所有的这些对象都能够表示在一个语义的空间里。

[word2vec]
---
    思路：用无监督的方法计算一个密集的低维词嵌入空间
    维度：抓住了特定的语义属性，比如性别
    模型：CBOW和skip-gram模型
    1）CBOW：CBOW 是 Continuous Bag-of-Words Model 的缩写，是一种根据上下文的词语预测当前词语的出现概率的模型。
    2）Skip-gram：Skip-gram只是逆转了CBOW的因果关系而已，即已知当前词语，预测上下文。
    都包含输入层，投影层和输出层，前者是上下文预测中间的词语，后者是知道当前的词预测上下文

```python
import logging
import gensim
from gensim.models import word2vec

# 设置输出日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 直接用gemsim提供的API去读取txt文件，读取文件的API有LineSentence 和 Text8Corpus, PathLineSentences等。
sentences = word2vec.LineSentence("/data4T/share/jiangxinyang848/textClassifier/data/preProcess/wordEmbdiing.txt")

# 训练模型，词向量的长度设置为200， 迭代次数为8，采用skip-gram模型，模型保存为bin格式
model = gensim.models.Word2Vec(sentences, size=200, sg=1, iter=8)  
model.wv.save_word2vec_format("./word2Vec" + ".bin", binary=True) 

# 加载bin格式的模型
wordVec = gensim.models.KeyedVectors.load_word2vec_format("word2Vec.bin", binary=True)

```