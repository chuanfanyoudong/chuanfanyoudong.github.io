﻿---
title: HMM
date: 2021-12-17
categories: 算法 #文章分類目錄 可以省略
tags: #文章標籤 可以省略
     - 自然语言处理
description: #你對本頁的描述 可以省略
---

# 隐马尔科夫（HMM）

## 简介

隐马尔科夫可以用于序列标注问题，比如分词过程，假设分词按照BEOS过程，在这里，隐马尔科夫就是从隐马尔科夫模型生成隐藏状态序列

## 基本参数

- 初始概率向量π：就是我一开始分别是B、E、O、S的概率如{”B“:0.6,“S”:0.4}

- 状态转移概率A：就是我从其中一个隐状态到达另一个隐藏状态的概率，如：当前为B，则下一个字符是E的概率是0.8

- 观测概率矩阵B：就是从当前隐藏状态生成当前的概率，也叫发射概率：如当前是B，生成字符“我”的概率是0.02

一个隐马尔科夫过程如下：
    
    - 1从初始概率生成当前隐藏状态
    - 2从隐藏状态生成当前词
    - 3从当前隐藏状态转移到下一个隐藏状态
    - 重复23，直至最后一个状态 
    
## 基本假设

- 齐次马尔科夫性假设：当前隐藏状态只依赖于前一个时刻的状态
- 观测独立性假设：任意时刻的观测结果只与该时刻的隐藏状态有关

## 三个基本问题

- 概率计算问题
    给定模型（π、A、B）和观测序列，计算该观测序列出现的概率
- 隐藏状态预测问题
    给定模型（π、A、B）和观测序列，计算最可能的隐藏状态序列
- 模型参数计算问题 
    给定观测序列，预测该模型的参数，是的生成该观测序列的概率最大
    
### 概率计算问题
    - 前向算法，就是假设有下一时刻的状态为i且观测结果是t的概率为前一时刻所有的隐藏状态每个单个的状态的概率乘上转移到i的概率（转移概率哦）之和
    - 然后与观测概率（发射概率）相乘(观测概率就是当前状态为i，观测序列为t的概率)
    - 例子：假设观测序列为o1,o2,o3,...,on
    - 定义a(t,i)为在o1,o2,o3...ot的观测序列下，状态为i
    - a(t+1,i)为在o1,o2,o3...ot+1的观测序列下，状态为i
    - 怎么求a(t+1,i)呢，求法就是a(t,j)*Aji对所有的j求和然后乘上Bi(t+1)
    - 依次迭代，求得结果

### 隐藏状态预测问题
    - 经典的维特比算法求解
        
### 模型参数计算问题
    - 监督学习，就是做统计，举个例子，当前状态为B，则下一个状态为E的概率是所有的当前为B的情况总和为分母，当前为B下一个状态为E的数量为分子，二者的商就是B->E的转移概率
    - 监督学习需要训练数据
    - 无监督学习，Baum-Welch方法（还没看懂）

## Trick

- 三阶马尔科夫并不很好的提高效果，倾向于生成新词


# 用一个分词的HMM的例子做个解释

## 任务：

**将“我来到苏州”分词**

## 理想结果

**【“我”，“来到”，“苏州”】**

## 定义参数

**要定义的参数主要有：状态参数、结果参数、初始化参数、转移概率、发射概率**

- 状态参数：这里就是每个字符的状态，我们采用简单的“BES“标记，如果一个字符作为一个词的开头则为B，如例子中的”来“，”苏“；如果一个字符作为一个词的结尾则为E，如例子中的”到“，”州“；如果一个字符作为一个单独词则为S，如例子中的”我“。所以状态参数为【”B“,"E","S"】

- 结果参数

- 初始化参数、就是在分词一开始的字符的状态参数：很明显不可能为E，因为E前边一定有B，所以我们可以统计海量语料中的第一个字符是作为单独词和词语的概率，用统计方法即可。在此，我们设计一个虚拟值：{”B“:0.6,"S":0.4},也就是第一个字作为B的概率是0.6，第一个字作为单独词的概率是0.4。

- 转移参数：就是上一个状态推导出下一个状态的概率，比如：已知上一个状态是”E“，则下一个字是”E“的概率是0.6，下一个状态是”S“的概率是0.4.所以我们设定转移参数是：

![image_1ct7m09th1vrc1fka1ocg1hlpdsc19.png-12.2kB][1]

举个例子：

    - 第二行第三列的意思是当前状态是B，下一状态是E的概率是1，也就是前一个状态如果是B，则下一个状态一定是E。很好理解，因为B是词语的开头，所以后面一定是E

    - 第三行第四列的意思是当前状态是E，下一状态是S的概率是1，也就是前一个状态如果是E，则下一个状态一定是S。

- 发射概率：就是在当前状态（比如”B“）的条件下，能够产生具体字符的概率，比如：在当前状态是B，那么产生”我“这个字符的概率是0.2，这也是基于统计的，就是统计所有的状态”B“后面的词，是”我”的概率就是
0.2



![image_1ct7s9ddu1me0n3a1rbj1ma3d4b1m.png-16.7kB][2]

举个例子：

    - 第二行第三列的意思是当前状态是B，下一个字符是来的概率是0.8
    - 第三行第四列的意思是当前状态是E，下一个字符是来的概率是0.5

## 维特比算法图解

**第一步**

根据初始化概率{“B”:0.6,"S":0.4,"E":0.0},可以得到如下图

![image_1ct7stn7t1djokus1a981v5f1oh923.png-69.5kB][3]

但是我们计算的不是这个值，我们要和发射概率相乘，因为我们要看的是第一个字符是“我“的概率，所以我们还要乘上发射概率。

所以第一个节点的状态：
    
    "B": B的初始化概率 * "B"到”我“的发射概率 = 0.6 * 0.4 = 0.24
    "E": E的初始化概率 * "E"到”我“的发射概率 = 0.0 * 0.1 = 0.00
    "S": S的初始化概率 * "S"到”我“的发射概率 = 0.4 * 0.8 = 0.32
![image_1ct7tbskpdsv1imvusd1n351eeg30.png-71.6kB][4]    

**第二步：**

![image_1ct7thvnr1cc1meg1fqjqg41mir4a.png-71.6kB][5]

我们知道了”我“，然后开始求”来“

我们将第一步得到的三个概率作为传递给”来“，然后根据发射概率求得第二部是每个状态的值：

**注意** 求解第二步的过程中，可能遇到多个结果，我们取最大值，比如：在第二步为B的情况有三种：

- 1、第一步为B，第二步为B, 此时概率为第一步为B的概率*B到B的转移概率*B到”来“的发射概率 = 0.24（上一步求得）* 0（根据转移矩阵）*0.8（根据发射概率矩阵求得） = 0

- 2、第一步为E，第二步为B，此时概率为第一步为E的概率*E到B的转移概率*B到”来“的发射概率 = 0（上一步求得）* 2（根据转移矩阵）*0.8（根据发射概率矩阵求得） = 0

- 3、第一步为S，第二步为B,此时概率为第一步为S的概率*S到B的转移概率*B到”来“的发射概率 = 0.32（上一步求得）* 0.8（根据转移矩阵）*0.4（根据发射概率矩阵求得） = 0.32*0.32 = 0.1024

**所以上述三个结果取最大值得到第二步为B的概率为0.1024, 路径是SB**
同理：
**所以上述三个结果取最大值得到第二步为E的概率为0.12,路径是BE**
**所以上述三个结果取最大值得到第二步为S的概率为0.0256，路径是SS**

**第三步：**

![image_1ct7uasv41msu75a2bo1qbi1h0354.png-74.8kB][6]

我们知道了”来“，然后开始求”到“

原理和上面一样，我直接放结果：

![image_1ct7unpjb56e16n01kfpucojs25h.png-80.5kB][7]

红色表示最优的路径

**第四步：**

![image_1ct7up4o61b51106v1jttmon1rbm6h.png-80.5kB][8]

我们知道了”到“，然后开始求”苏“
    
原理和上面一样，我直接放结果：

![image_1ct7vlb5o1lpd4811jt1gerppo98.png-84.2kB][9]

红色表示最优的路径

**第五步：**

![image_1ct7vlugu4991mju18aa1hf719gf9l.png-86.7kB][10]

我们知道了”苏“，然后开始求”州“

原理和上面一样，我直接放结果：

![image_1ct7vtgeh1dlr25g3f61npoff2a2.png-84.3kB][11]


红色表示最优的路径

**所以最优概率是0.0073728，所以最佳路径是SBEBE(图标红)**

## 结论

`上面就是一个简单的HMM实现的例子，其中求解的五步过程就是维特比算法`


  [1]: http://static.zybuluo.com/chuanfanyoudong/rndhslkqst9grmlw7cp6igxa/image_1ct7m09th1vrc1fka1ocg1hlpdsc19.png
  [2]: http://static.zybuluo.com/chuanfanyoudong/ajbpkxtsq2cw8jhk2j4om94v/image_1ct7s9ddu1me0n3a1rbj1ma3d4b1m.png
  [3]: http://static.zybuluo.com/chuanfanyoudong/hu0dfdd03hv2gprvde22ognc/image_1ct7stn7t1djokus1a981v5f1oh923.png
  [4]: http://static.zybuluo.com/chuanfanyoudong/rgjv6ktf8b3ojk3ekaenog4q/image_1ct7tbskpdsv1imvusd1n351eeg30.png
  [5]: http://static.zybuluo.com/chuanfanyoudong/d5wmk0vol91r16tqjptr91vi/image_1ct7thvnr1cc1meg1fqjqg41mir4a.png
  [6]: http://static.zybuluo.com/chuanfanyoudong/2ixhl5vzm6c4rujmo8wuyi2r/image_1ct7uasv41msu75a2bo1qbi1h0354.png
  [7]: http://static.zybuluo.com/chuanfanyoudong/m1kskhlj91fv4gpped3i9l0p/image_1ct7unpjb56e16n01kfpucojs25h.png
  [8]: http://static.zybuluo.com/chuanfanyoudong/biradjdr5hetfundnzad526s/image_1ct7up4o61b51106v1jttmon1rbm6h.png
  [9]: http://static.zybuluo.com/chuanfanyoudong/pcr9nhin8uz23le4pl6i6cun/image_1ct7vlb5o1lpd4811jt1gerppo98.png
  [10]: http://static.zybuluo.com/chuanfanyoudong/h9s4edk4fhmcfvxgqsrt93of/image_1ct7vlugu4991mju18aa1hf719gf9l.png
  [11]: http://static.zybuluo.com/chuanfanyoudong/j6a9vkvck5f5vipr9cvd2bkm/image_1ct7vtgeh1dlr25g3f61npoff2a2.png

