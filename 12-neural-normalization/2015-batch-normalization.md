# 批归一化（Batch Normalization）：数学推导 Tutorial

> 首次提出或经典年份：2015  
> 类别：神经网络确定性归一化  
> 本章目标：用小批次统计量标准化中间激活并学习恢复尺度  
> 先修知识：张量索引、均值方差、链式法则

## 1. 问题从哪里来

神经网络归一化的公式非常相似，真正区别在于均值和方差沿哪些轴计算。推导需要明确归一化集合、反向传播中统计量的依赖关系，以及可学习尺度参数的作用。

**批归一化（Batch Normalization）**要解决的具体问题是：用小批次统计量标准化中间激活并学习恢复尺度。本章不从最终公式开始，而是先定义数据对象和希望保留的关系，再逐步得到公式与算法。

### 1.1 本章将得到什么

- 推导前向性质
- 推导反向传播
- 说明最终公式与算法步骤之间的对应关系
- 用一个最小例子检查推导结果

## 2. 数据、符号与假设

一个批次的单通道激活 $x_1,\ldots,x_m$。可学习参数为 $\gamma,\beta$。

算法输出为：标准化并重新缩放的激活。

为了使上面的数学对象有定义，需要以下前提：

- 张量轴约定明确，$\varepsilon>0$。
- 训练与推理阶段使用的统计量规则按算法定义执行。

## 3. 建模前的基础数学

### 3.1 归一化的统一表达

对元素 $x_i$ 指定统计集合 $A(i)$，定义

$$
\mu_i=\frac1{|A(i)|}\sum_{j\in A(i)}x_j,
$$

$$
\sigma_i^2=\frac1{|A(i)|}\sum_{j\in A(i)}(x_j-\mu_i)^2.
$$

输出为

$$
y_i=\gamma_i\frac{x_i-\mu_i}{\sqrt{\sigma_i^2+\varepsilon}}+\beta_i.
$$

BatchNorm、LayerNorm 和 GroupNorm 的差别主要是 $A(i)$ 跨越哪些轴；RMSNorm 则不减均值。

## 4. 从定义到算法的完整推导

### A. 前向性质

对一个 mini-batch 的单个通道，

$$
\mu=\frac1m\sum_ix_i,
\qquad
\sigma^2=\frac1m\sum_i(x_i-\mu)^2,
$$

$$
\widehat x_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\varepsilon}},
\qquad y_i=\gamma\widehat x_i+\beta.
$$

当 $\varepsilon=0$ 时，直接求和得 $m^{-1}\sum_i\widehat x_i=0$，平方求和得 $m^{-1}\sum_i\widehat x_i^2=1$。

### B. 反向传播

令上游梯度 $g_i=\partial L/\partial y_i$。先有

$$
\frac{\partial L}{\partial\gamma}=\sum_ig_i\widehat x_i,
\qquad
\frac{\partial L}{\partial\beta}=\sum_ig_i.
$$

对 $x_i$ 同时考虑直接路径、均值路径和方差路径，整理得到

$$
\frac{\partial L}{\partial x_i}
=\frac{\gamma}{m\sqrt{\sigma^2+\varepsilon}}
\left[m g_i-\sum_jg_j-
\widehat x_i\sum_jg_j\widehat x_j\right].
$$

该式不能通过把均值和方差当常数得到；两条间接依赖必须保留。

## 5. 把推导压缩成最终公式

前一节给出了公式的来源。本节把最终计算所需的关系集中列出，便于实现时逐项对应。

### 5.1 1. 批均值与方差

$$
\mu_B=\frac1m\sum_{i=1}^mx_i,\qquad \sigma_B^2=\frac1m\sum_i(x_i-\mu_B)^2.
$$

### 5.2 2. 标准化

$$
\hat x_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\varepsilon}}.
$$

### 5.3 3. 仿射恢复

允许网络重新学习合适尺度和偏移。

$$
y_i=\gamma\hat x_i+\beta.
$$

### 5.4 4. 反向依赖

同一批次样本通过统计量耦合。

$$
\frac{\partial y_i}{\partial x_j}\ne0\quad\text{通常对 }i\ne j.
$$

### 5.5 5. 推理统计

推理阶段使用训练期间累计的总体近似。

$$
y=\gamma\frac{x-\mu_{\mathrm{run}}}{\sqrt{\sigma_{\mathrm{run}}^2+\varepsilon}}+\beta.
$$

## 6. 从公式到算法

**输入：** 一个批次的单通道激活 $x_1,\ldots,x_m$。可学习参数为 $\gamma,\beta$。
**输出：** 标准化并重新缩放的激活。

1. 按规范化轴计算批统计量。
2. 标准化激活。
3. 应用可学习仿射参数。
4. 更新运行均值和方差供推理使用。

上面的顺序直接来自前述推导：先计算定义算法状态所需的统计量，再求闭式解、执行递推或更新优化变量，最后按算法定义读取输出。这里没有加入独立于目标函数的额外规则。

## 7. 最小可复算例子

一个特征在 batch 中取值 $(1,3)$，均值 $\mu_B=2$，方差 $\sigma_B^2=1$。忽略 $\varepsilon$，标准化为 $(-1,1)$。若 $\gamma=2,\beta=0.5$，输出为

$$
y=(-1.5,2.5).
$$

训练时统计量由当前 batch 计算；推理时通常使用运行均值和方差，否则同一样本的输出会依赖同批其他样本。

## 8. 如何解释结果

本算法输出所表达的是“用小批次统计量标准化中间激活并学习恢复尺度”这一特定数学目标，而不是对数据全部结构的完整描述。解释结果时应直接回到目标函数、距离、概率或集合定义。

- 小批次统计噪声大。
- 训练与推理使用不同统计量。

## 9. 计算复杂度

$O(N)$ 于激活元素数。
