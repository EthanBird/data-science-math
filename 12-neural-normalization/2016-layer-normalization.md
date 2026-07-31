# 层归一化（Layer Normalization）：数学推导 Tutorial

> 首次提出或经典年份：2016  
> 类别：神经网络确定性归一化  
> 本章目标：在单个样本内部跨特征归一化激活  
> 先修知识：张量索引、均值方差、链式法则

## 1. 问题从哪里来

神经网络归一化的公式非常相似，真正区别在于均值和方差沿哪些轴计算。推导需要明确归一化集合、反向传播中统计量的依赖关系，以及可学习尺度参数的作用。

**层归一化（Layer Normalization）**要解决的具体问题是：在单个样本内部跨特征归一化激活。本章不从最终公式开始，而是先定义数据对象和希望保留的关系，再逐步得到公式与算法。

### 1.1 本章将得到什么

- 推导单样本内的均值与方差
- 推导仿射恢复
- 推导对正仿射输入的性质
- 说明最终公式与算法步骤之间的对应关系
- 用一个最小例子检查推导结果

## 2. 数据、符号与假设

单个样本某层激活 $x=(x_1,\ldots,x_H)$。

算法输出为：不依赖批大小的标准化激活。

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

### A. 单样本内的均值与方差

对隐藏向量 $x\in\mathbb R^H$，定义

$$
\mu=\frac1H\sum_{h=1}^Hx_h,
\qquad
\sigma^2=\frac1H\sum_h(x_h-\mu)^2.
$$

标准化量 $\widehat x_h=(x_h-\mu)/\sqrt{\sigma^2+\varepsilon}$。

若忽略 $\varepsilon$ 且 $\sigma^2>0$，则

$$
\frac1H\sum_h\widehat x_h
=\frac{\sum_hx_h-H\mu}{H\sigma}=0,
$$

$$
\frac1H\sum_h\widehat x_h^2
=\frac{\sum_h(x_h-\mu)^2}{H\sigma^2}=1.
$$

### B. 仿射恢复

$$
y_h=\gamma_h\widehat x_h+\beta_h.
$$

$\gamma,\beta$ 允许网络重新选择尺度与偏移；它们不影响归一化统计量的计算。

### C. 对正仿射输入的性质

令 $x'=ax+b\mathbf1$，$a>0$。则 $\mu'=a\mu+b$，且 $\sigma'^2=a^2\sigma^2$。当 $\varepsilon=0$：

$$
\frac{x'_h-\mu'}{\sigma'}
=\frac{a(x_h-\mu)}{a\sigma}
=\frac{x_h-\mu}{\sigma}.
$$

有限 $\varepsilon$ 时仅近似成立。LayerNorm 每个样本独立，不依赖批次其他样本。

## 5. 把推导压缩成最终公式

前一节给出了公式的来源。本节把最终计算所需的关系集中列出，便于实现时逐项对应。

### 5.1 1. 样本内统计量

$$
\mu=\frac1H\sum_{h=1}^Hx_h,\qquad \sigma^2=\frac1H\sum_h(x_h-\mu)^2.
$$

### 5.2 2. 标准化与仿射

$$
y_h=\gamma_h\frac{x_h-\mu}{\sqrt{\sigma^2+\varepsilon}}+\beta_h.
$$

### 5.3 3. 批独立性

统计量不跨样本。

$$
y_i\text{ depends only on }x_i.
$$

### 5.4 4. 平移与尺度

在 $\varepsilon$ 可忽略时，对正仿射缩放近似不变。

$$
\operatorname{LN}(ax+b\mathbf1)\approx\operatorname{LN}(x),\quad a>0.
$$

## 6. 从公式到算法

**输入：** 单个样本某层激活 $x=(x_1,\ldots,x_H)$。
**输出：** 不依赖批大小的标准化激活。

1. 对每个样本沿指定特征轴求均值和方差。
2. 标准化。
3. 应用逐特征可学习仿射参数。

上面的顺序直接来自前述推导：先计算定义算法状态所需的统计量，再求闭式解、执行递推或更新优化变量，最后按算法定义读取输出。这里没有加入独立于目标函数的额外规则。

## 7. 最小可复算例子

单个样本隐藏向量为 $h=(1,3)$。沿该样本的特征轴计算

$$
\mu=\frac{1+3}{2}=2,
\qquad
\sigma^2=\frac{(1-2)^2+(3-2)^2}{2}=1.
$$

忽略很小的 $\varepsilon$，标准化结果为

$$
\widehat h=\left(\frac{1-2}{1},\frac{3-2}{1}\right)=(-1,1).
$$

另一个样本会独立计算自己的均值和方差，因此输出不依赖同一 batch 中其他样本。

## 8. 如何解释结果

本算法输出所表达的是“在单个样本内部跨特征归一化激活”这一特定数学目标，而不是对数据全部结构的完整描述。解释结果时应直接回到目标函数、距离、概率或集合定义。

- 归一化轴必须与网络张量语义一致。

## 9. 计算复杂度

$O(N)$ 于激活元素数。
