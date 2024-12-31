# simple-nerf
A simplified implementation of  NeRF for  beginner.
NeRF保姆级教程

# 概述
本人将分享自己在学习NeRF过程中的经验。主要分为基础理论和代码实现。我将提供我认为非常棒的学习资料，其中每一个资料我都认真学习，确认其内容正确易懂。

注意！**一定要对着论文学习，遇到不懂的地方再去查找资料！**不要想着学完所有知识再去看论文！

# 理论篇
NeRF的实现主要分为三个大部分：由图像生成射线及其上的采样点、神经网络模型、体渲染。可以先看点视频大概了解一下NeRF。
https://www.youtube.com/watch?v=CRlN-cYFxTk

## 生成射线
数据集由图像和相机参数构成，NeRF神经网络模型的输入并不是图片本身，而是有图像生成的射线，像素点与一条射线是一一对应的。

首先，你需要了解一些图形学和线性代数有关的知识。

- 相机模型/小孔成像模型: https://zhuanlan.zhihu.com/p/356546894

- 相机的外参矩阵: https://www.bilibili.com/video/BV12u411G71A/?spm_id_from=333.337.search-card.all.click&vd_source=7e04c02519ee442d070207d7dd7b6dd6
注意，视频里的原理正确，但是矩阵形式有小错误

- 生成射线: https://www.youtube.com/watch?v=ujkec9KBnI8
这个视频的坐标系变换展示的很清晰

- 采样： 

这里有两种样点，一种是均匀采样得到的，一种是精细采样得到的（基于MLP的输出）

## MLP神经网络模型
这里我假设你已经有一些深度学习基础，能够简单的使用pytorch。如果你是纯小白，这里我推荐两本书。

先看鱼书（自认为最好的入门书籍）https://book.douban.com/subject/30270959/ 了解基本原理，后看沐神的动手学深度学习学习（有难度）怎么用pytorch

- 位置编码：https://zhuanlan.zhihu.com/p/623432244 讲的有深度，说白了就是把一个实数映射为一个高维向量。

- 网络模型：直接看论文即可，要注意图中的激活函数是看层前的箭头颜色，且层上的+拼接是指拼接在输出上，有一点confusing。

个人觉得NeRF的重点其实不在于这一部分。

- 逆变换采样

这里有两个MLP，一个粗网络一个精细网络，粗网络的结果评价空间中哪些位置比较重要。因为空间中有很多是空白的空气而非我们所关注的模型本身。在重要的地方多放置样点，不重要的地方少放置样点。

这部分说白了就是找反函数，需要一定的概率基础。

https://stephens999.github.io/fiveMinuteStats/inverse_transform_sampling.html#discrete_distributions

https://en.wikipedia.org/w/index.php?title=Inverse_transform_sampling&oldid=1244749103

nerf-pytorch的代码解读可以看 https://blog.csdn.net/susbsv/article/details/140291813


## 体渲染
这一部分如果没接触过比较难懂，有一点偏物理原理。模型的输出是一条射线上点的密度和颜色，体渲染是由此渲染出图像的过程。

- 光学模型：

参考文献[26]Max, N.: Optical models for direct volume rendering. IEEE Transactions on Visualization and Computer Graphics (1995) 的 Absorption only部分即可，后面的数值计算感觉和论文有差异。

https://zhuanlan.zhihu.com/p/595117334

- 数值计算：

https://zhuanlan.zhihu.com/p/595117334; 

http://arxiv.org/abs/2209.02417; 

或者看视频 【【较真系列】讲人话-NeRF全解（原理+代码+公式）】 https://www.bilibili.com/video/BV1CC411V7oq/?share_source=copy_web&vd_source=903a925cc99abfbbed2514de052b8cff

# 代码实现篇

**不推荐nerf-pytorch或者论文源代码**，过于冗长繁复，不适合新手。但是其中的部分的功能的函数可以学习一下。

可以选择性的看 https://book.douban.com/subject/37014639/ ，注意只要看代码，其他部分选择性的看一下就好。这本书原理讲解太简略，代码注释也不是很友好。

推荐 https://github.com/airalcorn2/pytorch-nerf （个人感觉变量名有点难懂）

或者看 https://colab.research.google.com/github/bmild/nerf/blob/master/tiny_nerf.ipynb （虽然是tensorflow的代码）

通过参考上面的代码，我自己也写了我实现的代码，写有中文注释，可以在colab上运行。https://colab.research.google.com/drive/1pO8PeNxHyuAvVAZjKPq-yvQnDYqqGppp