# TextRank

---
title: TextRank
date: 2022-01-14
categories: 算法 #文章分類目錄 可以省略
tags: #文章標籤 可以省略
     - 自然语言处理
description: #你對本頁的描述 可以省略

---

## 介绍

TextRank顾名思义就是对一篇文本中的词的重要性进行排序
在次之前，我们需要看一下PageRank, PageRank就是一个对网页排序的方法，其基本思想就是如果你这个网页有更多的其他网页链接到这里，那么这个网页就会越重要，这也比较符合人们的常识
而TextRank就是基于PageRank的一种变种，在一段文本中，你的这个词被更多的其他词“链接”到，那么我们认为这个词也会更重要，那么怎么确认词与词之间的“链接”关系呢？方法就是我们认为这个词的N-GRAM中的词会与这个词产生链接，在jieba中这个N用的是5，也就是说这个词与这个词附近的5个词会产生“链接”
我们还会考虑链接过来的词的重要性和个数，这个也很好理解，越重要的词或者“链接”过来的词越重要，那么分值越高。
下面我们公式化一下这个过程：

$$S(V_i) = (1 - d) + d \sum_{i,j}^n \frac{w_{ji}}{\sum_{k} w_{jk}}S(V_j)$$

其中$S(V_i)$就是我们要求的词i的重要性，$w_{ji}$就是词j和词i出现的一个统计，前面提到在距离为5的范围内，就会统计一次$w_{ji}$，所以在全文中有几次j和i出现的距离在5以内，那么其权重就是几，$\sum_{k} w_{jk}$是在求i的重要性过程中，不能单纯用j和i的统计次数，还要看j和i的关系在所有与j关联的词中的重要性，所以$\sum_{k} w_{jk}$就是所有与j关联的词中的统计数量之和の

> * 针对一条文本sentence，我们对这条文本切词,得到切词结果1,2,3...i...n
> * 假设每个词的重要性w_i，这个重要性也就是我们最终排序的依据。
> * 遍历句子的每个词，对每个词都和其后面的5（N-gram）个词做一个词对（i, i+1）,（i, i+2）,（i, i+3）,（i, i+4）,（i, i+5）这样对词i就有了所有与其关联的信息
> * 那么怎么算w_i也就是词i的重要性呢？



```python

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals
import sys
from operator import itemgetter
from collections import defaultdict
import jieba.posseg
from .tfidf import KeywordExtractor
from .._compat import *


class UndirectWeightedGraph:
    d = 0.85

    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, start, end, weight):
        # use a tuple (start, end, weight) instead of a Edge object
        self.graph[start].append((start, end, weight))
        self.graph[end].append((end, start, weight))

    def rank(self):
        ws = defaultdict(float)
        outSum = defaultdict(float)
        wsdef = 1.0 / (len(self.graph) or 1.0)
        for n, out in self.graph.items(): # self.graph 记录了这个所有的词和词搭配出现的count，搭配的定义在这里是在5个词范围内出现
            ws[n] = wsdef  # 给这个词一个默认分值
            outSum[n] = sum((e[2] for e in out), 0.0)  # 记录下这个词所有匹配过的词组的count和

        # this line for build stable iteration
        sorted_keys = sorted(self.graph.keys())  # 这个排序没看懂
        for x in xrange(10):  # 10 iters   遍历0-10
            for n in sorted_keys: # 对每一个key，也就是每一个单词
                s = 0
                for e in self.graph[n]:  # 对这个词的所有匹配结果
                    s += e[2] / outSum[e[1]] * ws[e[1]]  # 利用这个公式求s， 也就是说n这个词在e[1]的所有的词中的count比重越高，那么s的分值越高，然后利用这个重要程度对n的分值进行更新，
                    #  更新是延迟的，就是会出现平滑
                ws[n] = (1 - self.d) + self.d * s

        (min_rank, max_rank) = (sys.float_info[0], sys.float_info[3])

        for w in itervalues(ws):
            if w < min_rank:
                min_rank = w
            if w > max_rank:
                max_rank = w

        for n, w in ws.items():
            # to unify the weights, don't *100.
            ws[n] = (w - min_rank / 10.0) / (max_rank - min_rank / 10.0)

        return ws


class TextRank(KeywordExtractor):

    def __init__(self):
        self.tokenizer = self.postokenizer = jieba.posseg.dt
        self.stop_words = self.STOP_WORDS.copy()
        self.pos_filt = frozenset(('ns', 'n', 'vn', 'v'))
        self.span = 5

    def pairfilter(self, wp):
        return (wp.flag in self.pos_filt and len(wp.word.strip()) >= 2
                and wp.word.lower() not in self.stop_words)

    def textrank(self, sentence, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'), withFlag=False):
        """
        Extract keywords from sentence using TextRank algorithm.
        Parameter:
            - topK: return how many top keywords. `None` for all possible words.
            - withWeight: if True, return a list of (word, weight);
                          if False, return a list of words.
            - allowPOS: the allowed POS list eg. ['ns', 'n', 'vn', 'v'].
                        if the POS of w is not in this list, it will be filtered.
            - withFlag: if True, return a list of pair(word, weight) like posseg.cut
                        if False, return a list of words
        """
        self.pos_filt = frozenset(allowPOS)
        g = UndirectWeightedGraph()
        cm = defaultdict(int)
        words = tuple(self.tokenizer.cut(sentence))
        for i, wp in enumerate(words):
            if self.pairfilter(wp):
                for j in xrange(i + 1, i + self.span):
                    if j >= len(words):
                        break
                    if not self.pairfilter(words[j]):
                        continue
                    if allowPOS and withFlag:
                        cm[(wp, words[j])] += 1
                    else:
                        cm[(wp.word, words[j].word)] += 1

        for terms, w in cm.items():
            g.addEdge(terms[0], terms[1], w)
        nodes_rank = g.rank()
        if withWeight:
            tags = sorted(nodes_rank.items(), key=itemgetter(1), reverse=True)
        else:
            tags = sorted(nodes_rank, key=nodes_rank.__getitem__, reverse=True)

        if topK:
            return tags[:topK]
        else:
            return tags

    extract_tags = textrank


```




