[任务点]
---
前馈神经网络、网络层数、输入层、隐藏层、输出层、隐藏单元、9激活函数的概念。
感知机相关；定义简单的几层网络（激活函数sigmoid），递归使用链式法则来实现反向传播。
激活函数的种类以及各自的提出背景、优缺点。（和线性模型对比，线性模型的局限性，去线性化）
深度学习中的正则化（参数范数惩罚：L1正则化、L2正则化；数据集增强；噪声添加；early stop；Dropout层）、正则化的介绍。
深度模型中的优化：参数初始化策略；自适应学习率算法（梯度下降、AdaGrad、RMSProp、Adam；优化算法的选择）；batch norm层（提出背景、解决什么问题、层在训练和测试阶段的计算公式）；layer norm层。
FastText的原理。
利用FastText模型进行文本分类。
fasttext1 fasttext2 fasttext3 其中的参考

[神经网络]
---
深度前馈网络(deep feedforward network)，又名前馈神经网络或多层感知机(multilayer perceptron，MLP)，前馈的意思是指在这个神经网络里信息只是单方向的向前传播而没有反馈机制
前馈神经网络(feedforward neural network)是种比较简单的神经网络，只有输入层input layer (黄)、隐藏层hidden layer (绿)、输出层output layer
激活函数
神经网络神经元中，上层节点的输入值从通过加权求和后，到输出下层节点之前，还被作用了一个函数，这个函数就是激活函数(activation function)，作用是提供网络的非线性建模能力

[感知机]
---
感知机是二分类的线性分类模型，属于监督学习，用于分类。
感知机 = 一层的神经网络
好几层的感知机 = 神经网络
第一步：输入向量和权重进行点乘，假设叫做k
第二步：把所有的k都加起来得到加权和

[激活函数]
---
1.    定义：将输入信号的总和转换为输出信号的函数
作用：使之非线化
问题：为什么要使用激活函数？为什么要处理负值？
2.    为什么要使用非线性函数
因为使用线性函数的话，加深神经网络的层数就没有意义，发挥多层网络的优势
3.    输出层所用的激活函数    
回归问题可用恒等函数，二元分类问题可以使用 sigmoid函数，多元分类问题可以使用 softmax函数

tanh解决了sigmoid不是zero-centered输出问题，但是梯度消失(gradient vanishing)和幂运算的问题仍然存在
ReLU (Rectified Linear Unit) 现在最常用的激活函数：

[正则化]
---
过程：正则化是结构风险最小化策略的实现，是在经验风险上加一个正则化项或惩罚项。
正则化项一般是模型复杂度的单调递增函数，模型越复杂，正则化项就越大
1.   领回归 Ridge regression
①    增加惩罚项
②    λ无穷大，θ值逼近于0，导致underfitting
-----------两个作用：控制前部分拟合，同时是θ越小
引入正则项后，会使权重小的特征的权重损失得更多的，而对权重大的特征的权重相对来说损失的就少很多。
 L2 and L1 regularization
L1 Regularization (Lasso Regression Lasso回归)
L2 Regularization (Ridge Regression 岭回归)
岭回归是给原来的损失函数加上β \betaβ的平方作为惩罚项
L1与L2的区别
Lasso把不那么重要的特征的系数收缩为零。让这些特征说再见。所以，有巨多特征时，特征选择请找Lasso(产生稀疏模型)

[模型优化]
---
Momentum, NAG: address issue (i). Usually NAG > Momentum.
Adagrad, RMSProp: address issue (ii). RMSProp > Adagrad.
Adam, Nadam: address both issues, by combining above methods.

[FastText]
---
参考：https://blog.csdn.net/feilong_csdn/article/details/88655927
fastText是一个快速文本分类算法，与基于神经网络的分类算法相比有两大优点：
1、fastText在保持高精度的情况下加快了训练速度和测试速度
2、fastText不需要预训练好的词向量，fastText会自己训练词向量
3、fastText两个重要的优化：Hierarchical Softmax、N-gram
