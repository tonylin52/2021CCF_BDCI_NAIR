# 2021 CCF BDCI基于飞桨实现花样滑冰选手骨骼点动作识别-第1名方案
> 
## 项目描述
> CCF大数据与计算智能大赛基于飞桨实现花样滑冰选手骨骼点动作识别----新东方人工智能研究院队伍

## 硬件要求
1、由于模型基于3D卷积搭建，因此整体资源要求较高。
2、由于模型较多，为保证快速复现num_worker尽量不要设置为0，
3、因为num_worker不设置为0，训练过程中会占用大量的共享内存，因此共享内存目录（/dev/shm/）尽量分配多一些空间。若num_worker=0，则占用共享内存较小，但是训练速度降低较多。

## 项目整体架构
> 本算法主要由三个系列的模型集合组成XtSe系列、RSD系列模型、ResNet152系列模型。每个系列模型的数据与训练方式均有不同，整体算法架构如下图所示，详细说明在下文。
![avatar](1.png)

## 项目结构
> 一目了然的项目结构能帮助更多人了解，目录树以及设计思想都很重要~
```
| -- data
| -- XtSe
| -- Distill
| -- Res152
| -- test_B_data.npy
| -- work
| -- XtSe
    | -- *模型代码文件*
    | -- train.sh
    | -- test.sh
| -- Distill
| -- *模型代码文件*
    | -- train.sh
    | -- test.sh
| -- Res152
| -- *模型代码文件*
    | -- train.sh
    | -- test.sh
| -- Ensemble
    | -- *模型代码文件*
    | -- test.sh
| -- requirements.txt
| -- 花样滑冰比赛文档.docx
| -- .gitignore
| -- README.md

```
> 在work/路径下，有三个主要的模型对应的文件夹，其中XtSe/文件夹里面的是XtSe系列的模型，Distill/文件夹里面是RSD系列模型，Res152/文件夹里面是ResNet152系列模型，最终结果基于以上所有的模型的集成得到。


## 模型推理
> 相信你的Fans已经看到这里了，快告诉他们如何快速上手这个项目吧~  
A：在AI Studio上[运行本项目](https://aistudio.baidu.com/aistudio/usercenter)  
B：此处由项目作者进行撰写使用方式。


B榜推理使用
```

sh test.sh

```



## 模型训练
### 模型主要分三块


```

```
