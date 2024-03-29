---

title: 主题模型LDA
date: 2021-12-17
categories: 算法 #文章分類目錄 可以省略
tags: #文章標籤 可以省略
     - 自然语言处理
     - 主题模型
description: #你對本頁的描述 可以省略
---


## LDA

- 首先，LDA是一个词袋模型，与顺序无关。

- 几种分布

    - 伯努利分布
        
        ![image_1d3g9jrgp16521sdqaimff31d9f13.png-8.5kB][1]
    
    - Beta分布
        
        ![image_1d3g9ksl456eorkuch1vd9173t20.png-9.5kB][2]
        
    - 多项分布
    
        ![image_1d3g9nl9n5t618fnroh12ut19c43a.png-12.6kB][3]
    
    - 狄利克雷分布
    
        ![image_1d3g9vjlvtk1pem13b51p8e1bao44.png-12.3kB][4]
        
    - 四者关系
        
        ![image_1d3g9mptk1bmq1mc1ou2cd8116p2d.png-10.8kB][5]
        
        ![image_1d3g9vuat15bl14131ah7d6n1b9h4h.png-11.5kB][6]
        
    - lda
    
        ![image_1d3gaahas66d1ekbtnp1bluaiv4u.png-135.3kB][7]
        
        ![image_1d3gaauoj112mt66tbs10r81o25b.png-106.9kB][8]

- LDA默认一篇文档的生成过程是这样的，首先随机挑选一个主题，然后从这个主题中挑选一个词，然后依次生成每一个词，这样这篇文档就生成了。

- 主题的分布和词的分布都是符合狄利克雷分布的。

- 求解方法有吉布斯采样和变分推断法

- 吉布斯采样就是对每一个单词，先去掉，然后算出每一个类别的概率，然后根据这些类别的概率作为权重重新分配该单词的类别，然后更新参数

- LDA Gibbs采样算法的预测流程

    - 对应当前文档的每一个词，随机的赋予一个主题编号z
    - 重新扫描当前文档，对于每一个词，利用Gibbs采样公式更新它的topic编号。
    - 重复第2步的基于坐标轴轮换的Gibbs采样，直到Gibbs采样收敛。
    - 统计文档中各个词的主题，得到该文档主题分布。

- 

- lda数学八卦
    
    https://www.cnblogs.com/gasongjian/p/7631978.html


## 写在后面

- NLP任重道远！但是冲鸭！
- 欢迎关注Github：https://github.com/chuanfanyoudong/nlp_learn


  [1]: http://static.zybuluo.com/chuanfanyoudong/08tsv70aqw3yaud074dvf6ok/image_1d3g9jrgp16521sdqaimff31d9f13.png
  [2]: http://static.zybuluo.com/chuanfanyoudong/jbdk5yu84e57z0u1mxyunoye/image_1d3g9ksl456eorkuch1vd9173t20.png
  [3]: http://static.zybuluo.com/chuanfanyoudong/aubmmogao012mjh1agcr37zu/image_1d3g9nl9n5t618fnroh12ut19c43a.png
  [4]: http://static.zybuluo.com/chuanfanyoudong/j6kav1qou6cryj5w5ee4hd16/image_1d3g9vjlvtk1pem13b51p8e1bao44.png
  [5]: http://static.zybuluo.com/chuanfanyoudong/1rrd958mmmzc7i7kuaekq7cu/image_1d3g9mptk1bmq1mc1ou2cd8116p2d.png
  [6]: http://static.zybuluo.com/chuanfanyoudong/rws016sk8zltz3y29vyvycz7/image_1d3g9vuat15bl14131ah7d6n1b9h4h.png
  [7]: http://static.zybuluo.com/chuanfanyoudong/b93d8ijzfw6evv58f8rmei5t/image_1d3gaahas66d1ekbtnp1bluaiv4u.png
  [8]: http://static.zybuluo.com/chuanfanyoudong/ng8vb3tttisy88ksuxw9uhr2/image_1d3gaauoj112mt66tbs10r81o25b.png