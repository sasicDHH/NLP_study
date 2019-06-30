[任务点]
---
    1.  朴素贝叶斯
        朴素贝叶斯的原理
        利用朴素贝叶斯模型进行文本分类
        朴素贝叶斯1
    2.  SVM模型
        SVM的原理
        利用SVM模型进行文本分类
    3.  LDA主题模型
        pLSA、共轭先验分布
        LDA
        使用LDA生成主题特征，在之前特征的基础上加入主题特征进行文本分类
        LDA数学八卦 lda2 合并特征

[朴素贝叶斯]
---
    1.  基本假设：     ①特征之间相互独立  ②每个特征同等重要
            朴素贝叶斯法是基于贝叶斯定理与特征条件独立假设的分类方法
    2.    优点：    ①    易建立    ②    训练快    ③    多类别分类
    3.    用途：    文档分类---过滤论坛侮辱性词汇
    4.    实现方式：    ①伯努利模型（不考虑次数，只考虑01，即等权重） ②多项式模型
        模型：高斯模型    多项式模型    伯努利模型

(利用朴素贝叶斯模型进行文本分类)

    （1）分词：把一条记录进行分词，保存为word_list，所有的记录保存在data_list，所有的类别保存在class_list 
    （2）划分训练集和测试集 
    （3）统计词频：统计所有训练集的每个词出现的次数，即词频，放入all_words_dict字典中，对词频进行降序排序，保存为all_words_list 
    （4）停用词表：载入停用词表stopwords_set 
    （5）文本特征提取：选取n=1000个特征词（词频高到低依次选）保存在feature_words（选取all_words_list中（all_words_list中词频进行降序排序）排除出现在停用词表中的词，排除数字，并且要求词的长度为(1,5)） 
    （6）每条记录的表示（表示成长度为1000的特征词）：依次遍历feature_words中每个词，对于每次遍历的词word，如果在当前需要表示的记录中，则该记录的这一位为1，否则为0。即features = [1 if word in text_words else 0 for word in feature_words]，这里的text_words 表示一条记录。 
    （7）训练，预测：使用MultinomialNB进行训练，预测 
    （8）deleteN：删掉前面的deleteN个feature_words词，即最高频的deleteN词不作为特征词，而是从其排序后面的feature_words列表中选1000个特征词，重复以上步骤，得到另一个预测结果。deleteN的取值在某个范围时候，分类效果最好

[SVM模型]
---
    作用：分类、回归、异常检测
    优点：
        ①    分类效果好
        ②    计算开销小，无局部最优（NN），内存效率高
        ③    有正则化参数，减少过拟合
        ④    高维适用，维数>样本数也适用
    缺点：，
        ①    原始分类器不加修饰仅适用处理二类问题（one-versus-the-rest approach）
        ②    无概率估计（Logistic Regression）
        ③    参数难以解释
        ④    对于大数据集训练时间长
        ⑤    对参数调节和核函数的选择敏感
    原理推导:   
   [支持向量机通俗导论](http://blog.csdn.net/macyang/article/details/38782399/)
   
 (利用SVM模型进行文本分类)
 
    步骤
         1.爬取语料数据
         2.语料的处理
         3.抽取测试语料
         4.分词处理
         5.语料标注
         6.打乱语料
         7.特征提取（特征选择）
         8.向量化
         9.参数调优
         10.训练模型
         11.预测结果
         
[LDA主题模型]
---
    LDA是基于贝叶斯模型的
    先验分布 + 数据（似然）= 后验分布
    
    先验信息    
    在抽取样本X之前，人们对所要估计的未知参数θ所了解的信息，通常称为先验信息.
    先验分布
    对未知参数θ的先验信息用一个分布形式P(θ)来表示，此分布p(θ)称为未知参数θ的先验分布.(即在实验前通过已知信息知道的分布)
    共轭先验分布
    在贝叶斯概率理论中，如果后验概率P(θ|X)和先验概率P(θ)满足同样的分布律(形式相同，参数不同)。那么，先验分布和后验分布被叫做共轭分布，同时，先验分布叫做似然函数的共轭先验分布。
    
 （LDA）
 
    LDA（Latent Dirichlet Allocation）是一种文档主题生成模型，也称为一个三层贝叶斯概率模型，包含词、主题和文档三层结构。所谓生成模型，就是说，
    我们认为一篇文章的每个词都是通过“以一定概率选择了某个主题，并从这个主题中以一定概率选择某个词语”这样一个过程得到。文档到主题服从多项式分布，主题到词服从多项式分布
    
[代码]
---
1.读取和处理数据
```python
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary

# Create a corpus from a list of texts
common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]

# Train the model on the corpus.
lda = LdaModel(common_corpus, num_topics=10)
```

2.将文本转化为词袋模型
```python
rom gensim.corpora import Dictionary
dct = Dictionary(["máma mele maso".split(), "ema má máma".split()])
dct.doc2bow(["this", "is", "máma"])
[(2, 1)]
dct.doc2bow(["this", "is", "máma"], return_missing=True)
([(2, 1)], {u'this': 1, u'is': 1})
```

3.运用lda模型
```python
from gensim.models import LdaModel
lda = LdaModel(common_corpus, num_topics=10)
lda.print_topic(1, topn=2)
'0.500*"9" + 0.045*"10"
```
