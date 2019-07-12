[任务点]循环和递归神经网络
---

    1.  RNN的结构。递归神经网络 循环神经网络的提出背景、优缺点。    着重学习RNN的反向传播、RNN出现的问题（梯度问题、长期依赖问题）、BPTT算法。
    
    2.  双向RNN
    
    3.  LSTM、GRU的结构、提出背景、优缺点。
    针对梯度消失（LSTM等其他门控RNN）、梯度爆炸（梯度截断）的解决方案。
    Memory Network（自选）
    
    4.  Text-RNN的原理。
    利用Text-RNN模型来进行文本分类。
    
    5.  Recurrent Convolutional Neural Networks（RCNN）原理。
    利用RCNN模型来进行文本分类。

[RNN]
---
    1.  背景
    RNN通过每层之间节点的连接结构来记忆之前的信息，并利用这些信息来影响后面节点的输出。RNN可充分挖掘序列数据中的时序信息以及语义信息，这种在处理时序数据时比全连接神经网络和CNN更具有深度表达能力，RNN已广泛应用于语音识别、语言模型、机器翻译、时序分析等各个领域。
    
    2.  RNN的训练方法——BPTT算法
    BPTT（back-propagation through time）算法是常用的训练RNN的方法，其实本质还是BP算法，只不过RNN处理时间序列数据，所以要基于时间反向传播，故叫随时间反向传播。BPTT的中心思想和BP算法相同，沿着需要优化的参数的负梯度方向不断寻找更优的点直至收敛。综上所述，BPTT算法本质还是BP算法，BP算法本质还是梯度下降法，那么求各个参数的梯度便成了此算法的核心。  这里寻优的参数有三个，分别是U、V、W。与BP算法不同的是，其中W和U两个参数的寻优过程需要追溯之前的历史数据，参数V相对简单只需关注目前，那么我们就来先求解参数V的偏导数
    
    3.  RNN存在的问题
    i.  梯度消失的问题
    RNN网络的激活函数一般选用双曲正切，而不是sigmod函数，（RNN的激活函数除了双曲正切，RELU函数也用的非常多）原因在于RNN网络在求解时涉及时间序列上的大量求导运算，使用sigmod函数容易出现梯度消失，且sigmod的导数形式较为复杂。事实上，即使使用双曲正切函数，传统的RNN网络依然存在梯度消失问题。
    无论是梯度消失还是梯度爆炸，都是源于网络结构太深，造成网络权重不稳定，从本质上来讲是因为梯度反向传播中的连乘效应，类似于：0.99100=0.36
    0.99100=0.36，于是梯度越来越小，开始消失，另一种极端情况就是1.1100=13780    1.1100=13780。
    ii. 长期依赖的问题
    还有一个问题是无法“记忆”长时间序列上的信息，这个bug直到LSTM上引入了单元状态后才算较好地解决

    4.  RNN反向推导
   [循环神经网络(RNN)模型与前向反向传播算法](https://blog.csdn.net/anshuai_aw1/article/details/85163572)

[双向RNN]
---
    1.  定义
    Bidirectional RNN(双向RNN)假设当前t的输出不仅仅和之前的序列有关，并且 还与之后的序列有关，例如：预测一个语句中缺失的词语那么需要根据上下文进 行预测；Bidirectional RNN是一个相对简单的RNNs，由两个RNNs上下叠加在 一起组成。输出由这两个RNNs的隐藏层的状态决定。
    

[LSTM]长短时记忆网络
---
    1.  背景
    由于当间隔不断增大时，RNN会丧失学习到连接如此远的信息的能力；
    LSTM网络通过精妙的门控制将短期记忆与长期记忆结合起来，在一定程度上解决了梯度消失的问题。
    
    2.  LSTM结构
    用了三个门来解决梯度消失的问题
    忘记门层：
    第一步：丢弃信息
    其中sigmoid函数是确定旧的细胞状态的保留比例
    其中1 表示“完全保留”，0 表示“完全舍弃”
    输入门层：
    第二步：确定新信息
    tanh作为激活函数创新一个新的候选值C ̃_t，其中sigmoid函数是确定新的细胞状态的保留比例
    更新细胞状态：
    第三步：更新细胞状态
    根据忘记门层的sigmoid函数确定需要丢弃的信息，根据输入门层的sigmoid函数确定需要更新的信息
    确定输出值
    通过tanh激活函数处理细胞状态的得到一个在-1到1之间的数据，sigmoid函数确定输出部分的比例
    
    3.  LSTM的核心思想
    LSTM的关键在于细胞的状态整个(绿色的图表示的是一个cell)，和穿过细胞的那条水平线。
    细胞状态类似于传送带。直接在整个链上运行，只有一些少量的线性交互。信息在上面流传保持不变会很容易。
    若只有上面的那条水平线是没办法实现添加或者删除信息的。而是通过一种叫做 门（gates） 的结构来实现的。
    门 可以实现选择性地让信息通过，主要是通过一个 sigmoid 的神经层 和一个逐点相乘的操作来实现的。
    sigmoid 层输出（是一个向量）的每个元素都是一个在 0 和 1 之间的实数，表示让对应信息通过的权重（或者占比）。比如， 0 表示“不让任何信息通过”， 1 表示“让所有信息通过”。
    LSTM通过三个这样的本结构来实现信息的保护和控制。这三个门分别输入门、遗忘门和输出门。

[Text-RNN]
---
    1.  原理：
    单向RNN结构：
    其中权重矩阵U、V、W共享
    双向RNN结构：
    正向计算和反向计算不共享权值
    
    2.  代码
   ```python
#模型结构代码
#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 64      # 词向量维度
    seq_length = 600        # 序列长度
    num_classes = 10        # 类别数
    vocab_size = 5000       # 词汇表达小

    num_layers= 2           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    rnn = 'gru'             # lstm 或 gru

    dropout_keep_prob = 0.8 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 128         # 每批训练大小
    num_epochs = 10          # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard


class TextRNN(object):
    """文本分类，RNN模型"""
    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.rnn()

    def rnn(self):
        """rnn模型"""

        def lstm_cell():   # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout(): # 为每一个rnn核后面加一个dropout层
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("rnn"):
            # 多层rnn网络
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]  # 取最后一个时序输出作为结果

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#输出结果
Test Loss:    0.2, Test Acc:  94.17%
Precision, Recall and F1-Score...
              precision    recall  f1-score   support

          体育       1.00      0.98      0.99      1000
          财经       0.94      0.97      0.96      1000
          房产       0.99      1.00      0.99      1000
          家居       0.92      0.76      0.84      1000
          教育       0.91      0.94      0.92      1000
          科技       0.93      0.97      0.95      1000
          时尚       0.96      0.93      0.95      1000
          时政       0.86      0.94      0.90      1000
          游戏       0.96      0.97      0.96      1000
          娱乐       0.96      0.96      0.96      1000

   micro avg       0.94      0.94      0.94     10000
   macro avg       0.94      0.94      0.94     10000
weighted avg       0.94      0.94      0.94     10000

Confusion Matrix...
[[979   0   0   0   9   4   1   0   2   5]
 [  0 975   0   1   1   0   0  18   5   0]
 [  0   0 996   0   2   2   0   0   0   0]
 [  0  11  11 762  26  42  19 101  13  15]
 [  1  10   0   5 935  18   3  19   6   3]
 [  0   1   0   6   6 973   0   3   8   3]
 [  3   0   0  38  13   2 928   1   6   9]
 [  0  21   0   8  24   5   0 939   1   2]
 [  0   7   0   0   6   1   8   2 969   7]
 [  0  13   0   4   4   4   5   6   3 961]]
Time usage: 0:02:58

   ```