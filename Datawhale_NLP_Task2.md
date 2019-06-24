[知识点] 
---
    1.  基本文本处理技能
        1.1 分词的概念（分词的正向最大、逆向最大、双向最大匹配法）；
        1.2 词、字符频率统计；（可以使用Python中的collections.Counter模块，也可以自己寻找其他好用的库）
    2.  
        2.1 语言模型中unigram、bigram、trigram的概念；
        2.2 unigram、bigram频率统计；（可以使用Python中的collections.Counter模块，也可以自己寻找其他好用的库）
    3.  文本矩阵化：要求采用词袋模型且是词级别的矩阵化
    步骤有：
        3.1 分词（可采用结巴分词来进行分词操作，其他库也可以）；
        3.2 去停用词；构造词表。
        3.3 每篇文档的向量化。


[分词方法]
---
【基于字符串匹配】
1.    定义：机械分词方法，按照一定的策略将待分析的汉字串与一个“充分大的”机器词典中的词条进行配，若在词典中找到某个字符串，则匹配成功
2.    按扫描方向分：正向匹配和逆向匹配
按照不同长度优先匹配：    最大（最长）匹配和最小（最短）匹配；
按照是否与词性标注过程相结合：    单纯分词方法和分词与词性标注相结合的一体化方法。
3.    常用方法
（1）正向最大匹配法（从左到右的方向）；
（2）逆向最大匹配法（从右到左的方向）；
（3）最小切分（每一句中切出的词数最小）；
（4）双向最大匹配（进行从左到右、从右到左两次扫描）
评价：速度快，时间复杂度可以保持在O（n）,实现简单，效果尚可；但对歧义和未登录词处理效果不佳

【基于理解】
1.    定义：过让计算机模拟人对句子的理解
2.    原理：在分词的同时进行句法、语义分析，利用句法信息和语义信息来处理歧义现象
3.    结构：三个部分：分词子系统、句法语义子系统、总控部分
在总控部分的协调下，分词子系统可以获得有关词、句子等的句法和语义信息来对分词歧义进行判断，即它模拟了人对句子的理解过程
4.    评价：需要使用大量的语言知识和信息，试验阶段

【基于统计】
1.   定义：在给定大量已经分词的文本的前提下，利用统计机器学习模型学习词语切分的规律（称为训练），从而实现对未知文本的切分
2.    主要的统计模型有：N元文法模型（N-gram），隐马尔可夫模型（Hidden Markov Model ，HMM），最大熵模型（ME），条件随机场模型（Conditional Random Fields，CRF）
3.    常用方法：最大概率分词方法和最大熵分词方法
4.    操作：使用分词词典来进行字符串匹配分词，同时使用统计方法识别一些新词，即将字符串频率统计和字符串匹配结合
5.    评价：发挥匹配分词切分速度快、效率高的特点，又利用了无词典分词结合上下文识别生词、自动消除歧义的优点

[代码][参考](https://blog.csdn.net/yyy430/article/details/88117430)
1.  词、字符频率统计
```python
#coding=utf-8
import os
from collections import Counter
sumsdata=[]
for fname in os.listdir(os.getcwd()):
    if os.path.isfile(fname) and fname.endswith('.txt'):
        with open(fname,'r') as fp:
            data=fp.readlines()
            fp.close()
        sumsdata+=[line.strip().lower() for line in data]
cnt=Counter()
for word in sumsdata:
    cnt[word]+=1
cnt=dict(cnt)
for key,value in cnt.items():
    print(key+":"+str(value))
```
2.  jieba分词
```python
import jieba
 
# 全模式
text = "我来到北京清华大学"
seg_list = jieba.cut(text, cut_all=True)
print(u"[全模式]: ", "/ ".join(seg_list))
 
# 精确模式
seg_list = jieba.cut(text, cut_all=False)
print(u"[精确模式]: ", "/ ".join(seg_list))
 
# 默认是精确模式
seg_list = jieba.cut(text)
print(u"[默认模式]: ", "/ ".join(seg_list))
 
# 搜索引擎模式
seg_list = jieba.cut_for_search(text)
print(u"[搜索引擎模式]: ", "/ ".join(seg_list))
```
3.  去除停用词
```python
import jieba
 
#导入自定义词典
jieba.load_userdict('F:\system\Anaconda3\Lib\site-packages\jieba\mydict.txt')
 
# 去除停用词
stopwords = {}.fromkeys(['的', '包括', '等', '是'])
text = "故宫的著名景点包括乾清宫、太和殿和午门等。其中乾清宫非常精美，午门是紫禁城的正门。"
# 精确模式
segs = jieba.cut(text, cut_all=False)
final = ''
for seg in segs:
    if seg not in stopwords:
            final += seg
print (final)
 
seg_list = jieba.cut(final, cut_all=False)
print ("/ ".join(seg_list))
```
4.  构造词表
```python
def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)
 
    all_data = []
    for content in data_train:
        all_data.extend(content)
 
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')
```
5.  文档向量化
```python
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
 
 
# 读取停用词
def read_stopword(filename):
    stopword = []
    fp = open(filename, 'r')
    for line in fp.readlines():
        stopword.append(line.replace('\n', ''))
    fp.close()
    return stopword
 
 
# 切分数据，并删除停用词
def cut_data(data, stopword):
    words = []
    for content in data['content']:
        word = list(jieba.cut(content))
        for w in list(set(word) & set(stopword)):
            while w in word:
                word.remove(w)
        words.append(' '.join(word))
    data['content'] = words
    return data
 
 
# 获取单词列表
def word_list(data):
    all_word = []
    for word in data['content']:
        all_word.extend(word)
    all_word = list(set(all_word))
    return all_word
 
 
# 计算文本向量
def text_vec(data):
    count_vec = CountVectorizer(max_features=300, min_df=2)
    count_vec.fit_transform(data['content'])
    fea_vec = count_vec.transform(data['content']).toarray()
    return fea_vec
 
 
if __name__ == '__main__':
    data = pd.read_csv('./cnews/test.txt', names=['title', 'content'], sep='\t')  # (10000, 2)
 
    stopword = read_stopword('./cnews/stopword.txt')
    data = cut_data(data, stopword)
 
    fea_vec = text_vec(data)
    print(fea_vec)
```
[n-gram语法模型]
---

    1.  Unigram（一元）：当前词出现的概率为自身词频
    2.  Bigram（二元）：只与前面一个词相关
    3.  Trigram（三元）：只与前面两个词相关