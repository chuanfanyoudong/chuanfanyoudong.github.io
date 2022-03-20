---
title: Transformer解释
date: 2021-12-17
categories: 算法 #文章分類目錄 可以省略
tags: #文章標籤 可以省略
     - 自然语言处理
description: #你對本頁的描述 可以省略

---
 <meta name="referrer" content="no-referrer" />

[toc]

- 本文基本翻译自：http://jalammar.github.io/illustrated-transformer
- http://jalammar.github.io/illustrated-transformer/
- 论文地址：https://arxiv.org/abs/1706.03762

## 概述

- encoder和decode都包括六层，每层相同，每层之间也不共享权重

![image_1d4pqb2rtjes1abv1qjaakq1s8n9.png-105kB][1]

- 每一层encoder的结构如下


![image_1d4pqc2471tccakd6h47pd1ek7m.png-26kB][2]

- self_attention:帮助encoder在解码一个字的时候关注到句子的其余词语

- self_attention的输出会送到feed_forward网络，

- 每一层decoder的结构如下
 
![image_1d4pqk58d27o1d3fakn1i451maa13.png-32.7kB][3]

- decoder层有encoder层的两层结构（self_attention和feed_foreard)并且在两层中间有一个encoder-decoder-attention层,这一层能够帮助关注输入和输出的注意力机制

## 用向量描述一下

- 首先我们把输入的所有词转换成词向量，如下图示：

![image_1d4pqrqj56qiesocmj1nrsgth20.png-12.8kB][4]

- 只有最开始输入的encoder的维度是词向量的维度，剩下的所有的encoder的维度都是隐藏层维度，在这里可以是512（这个维度是可以人为设定的）

- 下面进入encoder环节

![image_1d4pr1cah15pq8lgc9rkn4ifh2d.png-54.5kB][5]

- self-attention不可以并行，但是feed-forward可以并行

##  encoding过程

- 图示：
![image_1d4prae4v1rcu1jqo1bss17411ddt2q.png-74.9kB][6]

### selof_attention

- 不要被sdelf_attention这个概念唬住，其实在这边论文出来之前，我们就已经知道self_attention啦
- 我们打算翻译下面这句话：
    ![image_1d4pre3i69bhku24t91ft815g537.png-4kB][7]
- 当我们encoder“it”这个词的时候，self_attention可以让“it”和“animal”这个词关联起来
- 对每个词都这样就会使得encoder效果更好
- 这里self_attention是用来和RNN做对应的，都可以在处理这个词的时候将其他词的向量融合进去，下面图示
 ![image_1d4prqrb12k2ahh12fhvcnpmk3k.png-46.3kB][8]
- 一个测试transfor的脚本链接：https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb

### self_attention的细节

- 下图示：
![image_1d4psl2c71go8t1mbqa1be415d741.png-44.5kB][9]

- 对每一个输入的单词向量， 会有一个query向量，一个key向量，一个value向量
- 这三个向量的来历：在我们进行到“thinking”这个词的时候，我们会计算“thinking”这个词需要的每个词的权重， 就是encoder “thinging”这个词的时候，我们会对其他词的关注程度，这里的计算方法，就是“thinging”这个词的q与所有词的k进行点乘，作为权重得分。如下图所示：
- ![image_1d4pst1p2161k1hj887t14abftq4e.png-29kB][10]
-下一步就是除以key向量的维度的开方，比如key的维度是64，那么要除以64的开方，也就是8，最后进行softmax，得到0.88和0.12，这样做是为了有更平缓的梯度。
- 这一步将决定没个词的权重，诚然本身这个词（“thinking”）的权重会最大，但是其余此有时候也会很有用
![image_1d4pt3gru858tpe3q18bi16k44r.png-48.2kB][11]
- 下一步就是将上一步softmax的得分与value向量相乘，求和输出，如下图示：
![image_1d4ptgg1mibs1vlsdrc1p7b1bhp71.png-57.9kB][12]

### self_attention的矩阵算法

- query,key,value怎么来的？也是算出来的看图：
- ![image_1d4ptoq651bdkaqp9g814qt1ijb7e.png-25kB][13]
- 这里假设只有两个词，词向量维度是4，所以X的维度是2*4，所以乘上WQ（4*3,3是query的维度）就求出query矩阵，同理：可以求出key矩阵，value矩阵。
-![image_1d4ptv8521e0sk1c1svs6m4mqn7r.png-20.4kB][14]
- 解释一下上图矩阵的求法，Q的每一行代表每一个词，与K的每一列相乘，代表当前Q对k的当前列的权重，所以乘出来是一个2*2
- 其中点第一行第一列的点代表第一个词的query与第一个词的key的点乘
- 第一行第二列代表第一个词的query与第二个词的key的点乘
- 第二行第一列代表第二个词的query与第一个词的key的点乘
- 第二行第二列代表第二个词的query与第二个词的key的点乘
- 然后对每一行进行softmax，就得到了权重，然后与value矩阵相乘，就得到了最终的结果
- 2*2矩阵的第一行与value矩阵的第一列相乘，代表着第一个词的输出value向量在第一个维度上的加权输出。其余类似

### 多头注意力

- 虽然self_attention使得输出结果有了其他词的融合，但是主要还是这个词本身。
- 给了输出更大的表示向量空间

- 如图所示为：图中为有两个头

![image_1d4pun7oc1cck1tl3qi78n1imb8d.png-58.2kB][15]

- 每个头中的三个权重矩阵互不干涉，那么最终我们会得到8个（论文中有8个头数量）输出矩阵。
![image_1d4puqgki1dlh68e1j611st48q.png-52.2kB][16]

- 然后我们把八个矩阵拼起来再和另一个矩阵相乘，得到最终输出：

![image_1d4puspcj1p8m1gh2f0gp297f497.png-69.2kB][17]

- 所有的过程合并起来就是下图：

![image_1d4puuasenl81p5j15r15o71u57a4.png-131.1kB][18]


### 编码序列的位置信息

- 上文所有的过程都没有用到位置信息，我们记录了位置向量和输入的向量结合，如下图：
![image_1d4q0fd93gfhhfg1dicdqert2o.png-72.9kB][19]

- 就是两个向量对应位加和：
![image_1d4q0gs1tcp2sf6aai14nvdau35.png-35.3kB][20]
- 下图是一个20个词的512维度的位置向量计算方法， 中间有断层，是因为维度的前半部分和后半部分使用了两个函数
![image_1d4q0rh866gv1mdv5tfgop1ttf42.png-80.8kB][21]

### 残差信息

每一个头（8个头中的一个）的self_attention都要经历一次normalization。
![image_1d4q1378418k61crpo5k1tlnrda4v.png-75.6kB][22]

### 两层的结构如下：

![image_1d4q155u21oma75n17o7hn84rg5c.png-171.7kB][23]

## decoder部分

- 解码部分会用到encoder阶段生成的key矩阵和value矩阵。一次一个的生成词。
- 解码部分的只能输入要翻译的词前面的所有的词，因为实际情况下，你不可能知道后面的词语，所以会有一个遮蔽矩阵。
- encoder-decoder 矩阵的query矩阵使用的是decoder的query，而key矩阵和value矩阵使用的是encoder部分的矩阵。

## 最终的全连接与softmax

- 这个不多说了，看图
- ![image_1d4q1st4jbk4ic2bl84391jvj5p.png-72.7kB][24]
- 简单说就是decoder的输出，会经过全连接生成1*词表大小的矩阵，然后softmax选出最大值。

## 训练过程的复盘

## 损失函数——loss function

- 交叉熵

### 预测的时候用到了 beam search方法

- https://zhuanlan.zhihu.com/p/36029811?group_id=972420376412762112
- 主要思想就是翻译当前字符的时候，考虑到了上一字符的结果过，每次维护N个最佳路线，最终输出。


  [1]: http://static.zybuluo.com/chuanfanyoudong/9q7tw5tixorn3rm13m4q6u7t/image_1d4pqb2rtjes1abv1qjaakq1s8n9.png
  [2]: http://static.zybuluo.com/chuanfanyoudong/s6tfvmopezmjub33wgcr4b61/image_1d4pqc2471tccakd6h47pd1ek7m.png
  [3]: http://static.zybuluo.com/chuanfanyoudong/dhld92snu1ehr2g706wa1dwi/image_1d4pqk58d27o1d3fakn1i451maa13.png
  [4]: http://static.zybuluo.com/chuanfanyoudong/p2q828vx0wauoixlwnp5y16j/image_1d4pqrqj56qiesocmj1nrsgth20.png
  [5]: http://static.zybuluo.com/chuanfanyoudong/izr55upv6tosswq4sqf0ngf4/image_1d4pr1cah15pq8lgc9rkn4ifh2d.png
  [6]: http://static.zybuluo.com/chuanfanyoudong/xug97j0rk49ohz9k3um8q0g8/image_1d4prae4v1rcu1jqo1bss17411ddt2q.png
  [7]: http://static.zybuluo.com/chuanfanyoudong/2r73r3zsb9c12kpjusif2zlk/image_1d4pre3i69bhku24t91ft815g537.png
  [8]: http://static.zybuluo.com/chuanfanyoudong/t3uimub3nfvk1paovimcjc41/image_1d4prqrb12k2ahh12fhvcnpmk3k.png
  [9]: http://static.zybuluo.com/chuanfanyoudong/emtf9m6yhassaygunv0soc67/image_1d4psl2c71go8t1mbqa1be415d741.png
  [10]: http://static.zybuluo.com/chuanfanyoudong/my0nmxu1ndg0900sl7w1tjyt/image_1d4pst1p2161k1hj887t14abftq4e.png
  [11]: http://static.zybuluo.com/chuanfanyoudong/ck9qky0epb5dc67bksh1p3df/image_1d4pt3gru858tpe3q18bi16k44r.png
  [12]: http://static.zybuluo.com/chuanfanyoudong/a98fb6i8du6pef8da63fj51t/image_1d4ptgg1mibs1vlsdrc1p7b1bhp71.png
  [13]: http://static.zybuluo.com/chuanfanyoudong/ytraxnvimest7p5kmglhek3i/image_1d4ptoq651bdkaqp9g814qt1ijb7e.png
  [14]: http://static.zybuluo.com/chuanfanyoudong/34es21yw8swxcpda5i6x57a1/image_1d4ptv8521e0sk1c1svs6m4mqn7r.png
  [15]: http://static.zybuluo.com/chuanfanyoudong/fn4wfv5bsanp3e9wmu5tz7cl/image_1d4pun7oc1cck1tl3qi78n1imb8d.png
  [16]: http://static.zybuluo.com/chuanfanyoudong/9qm0ve87n68k85m43dxyahih/image_1d4puqgki1dlh68e1j611st48q.png
  [17]: http://static.zybuluo.com/chuanfanyoudong/b3z6otq34mcfpa82v9a36bg0/image_1d4puspcj1p8m1gh2f0gp297f497.png
  [18]: http://static.zybuluo.com/chuanfanyoudong/ok5ec4w07zgrzk4i0d25jwvj/image_1d4puuasenl81p5j15r15o71u57a4.png
  [19]: http://static.zybuluo.com/chuanfanyoudong/kiodb6kb4ow1r5p1du6vwu5q/image_1d4q0fd93gfhhfg1dicdqert2o.png
  [20]: http://static.zybuluo.com/chuanfanyoudong/wcp6ocl4vs5bwa287wdr5nse/image_1d4q0gs1tcp2sf6aai14nvdau35.png
  [21]: http://static.zybuluo.com/chuanfanyoudong/92zunmx5o3d5k6yqbj3v7ly2/image_1d4q0rh866gv1mdv5tfgop1ttf42.png
  [22]: http://static.zybuluo.com/chuanfanyoudong/r0e6jj7w5m4stfv3cwx8uyou/image_1d4q1378418k61crpo5k1tlnrda4v.png
  [23]: http://static.zybuluo.com/chuanfanyoudong/m6aezsoro4p99gzwuri48y64/image_1d4q155u21oma75n17o7hn84rg5c.png
  [24]: http://static.zybuluo.com/chuanfanyoudong/xzg1b3zvat4jq5fuddtu9azg/image_1d4q1st4jbk4ic2bl84391jvj5p.png