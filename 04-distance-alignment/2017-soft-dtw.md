# Soft-DTW（Soft Dynamic Time Warping）：数学推导 Tutorial

> 首次提出或经典年份：2017  
> 类别：距离、核与对齐  
> 本章目标：用平滑最小算子把 DTW 变为可微损失  
> 先修知识：度量空间、二次型、动态规划

## 1. 问题从哪里来

很多学习算法最终只接收样本之间的距离或相异度。因此，距离如何定义就等价于规定算法如何理解“相似”。对时间序列还要同时选择配对路径，问题会从普通距离计算变成受约束的动态规划。

**Soft-DTW（Soft Dynamic Time Warping）**要解决的具体问题是：用平滑最小算子把 DTW 变为可微损失。本章不从最终公式开始，而是先定义数据对象和希望保留的关系，再逐步得到公式与算法。

### 1.1 本章将得到什么

- 推导softmin 极限
- 推导路径 Gibbs 分布与梯度
- 说明最终公式与算法步骤之间的对应关系
- 用一个最小例子检查推导结果

## 2. 数据、符号与假设

局部代价矩阵 $\Delta_{ij}=d(x_i,y_j)$，平滑参数 $\gamma>0$。

算法输出为：平滑对齐损失与软对齐梯度。

为了使上面的数学对象有定义，需要以下前提：

- 先固定每个变量的尺度、缺失处理和距离定义。
- 动态对齐的边界条件和允许步长属于算法定义的一部分。

## 3. 建模前的基础数学

### 3.1 距离与相异度

严格度量 $d$ 需要满足非负性、同一性、对称性和三角不等式：

$$
d(x,z)\le d(x,y)+d(y,z).
$$

算法中使用的代价不一定满足全部条件。时间对齐代价依赖一条路径，通常应先称为相异度，除非已经单独证明度量性质。

### 3.2 二次型距离

若 $W\succeq0$，则

$$
d_W^2(x,y)=(x-y)^\top W(x-y).
$$

取 $W=A^\top A$，有

$$
d_W^2(x,y)=\|A(x-y)\|_2^2,
$$

所以加权距离等价于先线性变换坐标，再计算欧氏距离。

## 4. 从定义到算法的完整推导

### A. softmin 极限

设 $m=\min_k a_k$。则

$$
\sum_ke^{-a_k/\gamma}=e^{-m/\gamma}\sum_ke^{-(a_k-m)/\gamma}.
$$

取负对数：

$$
\operatorname{softmin}_\gamma(a)
=m-\gamma\log\sum_ke^{-(a_k-m)/\gamma}.
$$

括号内至少有一个 $1$，至多为 $K$，故

$$
m-\gamma\log K\le\operatorname{softmin}_\gamma(a)\le m.
$$

令 $\gamma\downarrow0$，夹逼定理给出 softmin 收敛到最小值。

### B. 路径 Gibbs 分布与梯度

Soft-DTW 可写为

$$
S(\Delta)=-\gamma\log\sum_\pi e^{-C_\pi(\Delta)/\gamma}.
$$

对某个局部代价 $\Delta_{ij}$ 求导，并使用链式法则：

$$
\frac{\partial S}{\partial\Delta_{ij}}
=\frac{\sum_\pi e^{-C_\pi/\gamma}
\frac{\partial C_\pi}{\partial\Delta_{ij}}}
{\sum_\pi e^{-C_\pi/\gamma}}.
$$

因为路径代价是所经过单元的和，

$$
\frac{\partial C_\pi}{\partial\Delta_{ij}}=\mathbf1\{(i,j)\in\pi\}.
$$

归一化权重正是 Gibbs 路径概率，所以梯度等于该单元被随机路径经过的概率。

## 5. 把推导压缩成最终公式

前一节给出了公式的来源。本节把最终计算所需的关系集中列出，便于实现时逐项对应。

### 5.1 1. 软最小

$$
\operatorname{softmin}_\gamma(a_1,\ldots,a_K)=-\gamma\log\sum_{k=1}^K e^{-a_k/\gamma}.
$$

### 5.2 2. 平滑动态规划

$$
R_{ij}=\Delta_{ij}+\operatorname{softmin}_\gamma(R_{i-1,j},R_{i,j-1},R_{i-1,j-1}).
$$

### 5.3 3. 极限

当 $\gamma\to0^+$ 时恢复普通最小值。

$$
\lim_{\gamma\to0^+}\operatorname{softmin}_\gamma(a)=\min_ka_k.
$$

### 5.4 4. 路径分布

目标等于对所有路径代价的 log-sum-exp。

$$
\operatorname{sDTW}_\gamma(x,y)=-\gamma\log\sum_{\pi}e^{-C(\pi)/\gamma}.
$$

### 5.5 5. 梯度

反向动态规划给出每个局部代价被使用的软期望次数。

$$
\frac{\partial\operatorname{sDTW}}{\partial\Delta_{ij}}=\mathbb E_{\pi\sim p_\gamma}[\mathbf1[(i,j)\in\pi]].
$$

## 6. 从公式到算法

**输入：** 局部代价矩阵 $\Delta_{ij}=d(x_i,y_j)$，平滑参数 $\gamma>0$。
**输出：** 平滑对齐损失与软对齐梯度。

1. 构造局部代价。
2. 用稳定 log-sum-exp 填充前向表。
3. 从终点执行反向递推。
4. 链式法则传播到序列。

上面的顺序直接来自前述推导：先计算定义算法状态所需的统计量，再求闭式解、执行递推或更新优化变量，最后按算法定义读取输出。这里没有加入独立于目标函数的额外规则。

## 7. 最小可复算例子

若某格有三条前驱累计代价 $(1,2,4)$，平滑参数 $\gamma=1$，SoftMin 为

$$
-\log(e^{-1}+e^{-2}+e^{-4})\approx0.651.
$$

硬最小值为 $1$。平滑值同时受到三条路径影响，因此对每条前驱代价都有非零梯度；当 $\gamma\downarrow0$ 时，该值趋于 $1$。

## 8. 如何解释结果

本算法输出所表达的是“用平滑最小算子把 DTW 变为可微损失”这一特定数学目标，而不是对数据全部结构的完整描述。解释结果时应直接回到目标函数、距离、概率或集合定义。

- 它不一定满足距离公理。
- $\gamma$ 决定对齐路径分布的平滑程度。

## 9. 计算复杂度

时间和空间 $O(nm)$。
