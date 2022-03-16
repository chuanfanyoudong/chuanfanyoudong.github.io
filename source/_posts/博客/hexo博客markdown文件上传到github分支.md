---
title: hexo博客markdown文件上传到github分支
date: 2020-02-02
categories: "博客"
tags: #文章標籤 可以省略
     - hexo
description: hexo博客markdown文件上传到github分支

---

## 综述

使用hexo结合github博客需要在自己的github上建立一个和自己的github用户名相同的库，如：
**https://github.com/chuanfanyoudong/chuanfanyoudong.github.io**
但是发现在这个库的master分支只是存储了生成的html代码，没有存储原始的markdown文件，但是这样不利于我们的markdown文件的安全，把他放在github上是比较安全的，目前主要有两个方法：
* 1.新建一个其他库，用来存放markdown文件
* 2.在当前库下建立一个其他分支，存储markdown文件

在这里本着不能太罗嗦的原则，我选择了第二种方案，基本上需要下面几步：
* 1.新建一个分支source（名字可自取），用来存储markdown文件 
    
    

* 2.将**chuanfanyoudong.github.io**的默认分支有改为source
    
    ![更改默认分支](../../images/blog/20200202-02.jpg)

* 3.线下配置
    
    * 你需要利用git clone 把数据拉下来，注意这时候拉下来只是转好的html代码，也就是只是博客内容里面public文件夹下面的内容
    ![一开始拉下来的代码](../../images/blog/20200202-03.jpg)
    * 然后你把目录覆盖成博客里面的内容就好了
    ![一开始拉下来的代码](../../images/blog/20200202-04.jpg)
    * git add .
    * git commit -m "commit hexo markdown file"
    * git push origin source

 