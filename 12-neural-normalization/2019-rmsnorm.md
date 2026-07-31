# RMSNorm（Root Mean Square Layer Normalization）：数学推导 Tutorial

> 首次提出或经典年份：2019  
> 类别：神经网络确定性归一化  
> 本章目标：仅按均方根缩放激活，保留均值信息并简化层归一化  
> 先修知识：张量索引、均值方差、链式法则

## 1. 问题从哪里来

神经网络归一化的公式非常相似，真正区别在于均值和方差沿哪些轴计算。推导需要明确归一化集合、反向传播中统计量的依赖关系，以及可学习尺度参数的作用。

**RMSNorm（Root Mean Square Layer Normalization）**要解决的具体问题是：仅按均方根缩放激活，保留均值信息并简化层归一化。本章不从最终公式开始，而是先定义数据对象和希望保留的关系，再逐步得到公式与算法。

### 1.1 本章将得到什么

- 推导均方根尺度
- 推导正缩放不变性
- 推导与 LayerNorm 的区别
- 说明最终公式与算法步骤之间的对应关系
- 用一个最小例子检查推导结果

## 2. 数据、符号与假设

激活向量 $x\in\mathbb R^H$，可学习尺度 $g\in\mathbb R^H$。

算法输出为：均方根受控但均值未被移除的激活。

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

### A. 均方根尺度

$$
\operatorname{RMS}(x)
=\sqrt{\frac1H\sum_{h=1}^Hx_h^2+\varepsilon}.
$$

因为平方和非负且 $\varepsilon>0$，分母严格为正。

输出为

$$
y_h=g_h\frac{x_h}{\operatorname{RMS}(x)}.
$$

### B. 正缩放不变性

忽略 $\varepsilon$，对 $a>0$：

$$
\operatorname{RMS}(ax)
=\sqrt{\frac1H\sum_ha^2x_h^2}
=a\operatorname{RMS}(x).
$$

因此

$$
\frac{ax}{\operatorname{RMS}(ax)}
=\frac{x}{\operatorname{RMS}(x)}.
$$

有 $\varepsilon$ 时等式不再精确。

### C. 与 LayerNorm 的区别

LayerNorm 先减均值，RMSNorm 不减：

$$
\frac{x-\bar x\mathbf1}{\sqrt{H^{-1}\sum_h(x_h-\bar x)^2}}
\ne
\frac{x}{\sqrt{H^{-1}\sum_hx_h^2}}
$$

除非均值和结构满足特殊条件。因此 RMSNorm 不具有平移不变性。

## 5. 把推导压缩成最终公式

前一节给出了公式的来源。本节把最终计算所需的关系集中列出，便于实现时逐项对应。

### 5.1 1. 均方根

$$
\operatorname{RMS}(x)=\sqrt{\frac1H\sum_{h=1}^Hx_h^2+\varepsilon}.
$$

### 5.2 2. 缩放

$$
y_h=g_h\frac{x_h}{\operatorname{RMS}(x)}.
$$

### 5.3 3. 正尺度不变性

$$
\frac{ax}{\operatorname{RMS}(ax)}=\frac{x}{\operatorname{RMS}(x)},\qquad a>0.
$$

### 5.4 4. 与 LayerNorm 区别

RMSNorm 不减去均值。

$$
\operatorname{RMSNorm}(x)\ne\operatorname{LN}(x)\quad\text{when }\bar x\ne0.
$$

## 6. 从公式到算法

**输入：** 激活向量 $x\in\mathbb R^H$，可学习尺度 $g\in\mathbb R^H$。
**输出：** 均方根受控但均值未被移除的激活。

1. 沿特征轴计算均方根。
2. 逐元素除以均方根。
3. 乘可学习尺度。

上面的顺序直接来自前述推导：先计算定义算法状态所需的统计量，再求闭式解、执行递推或更新优化变量，最后按算法定义读取输出。这里没有加入独立于目标函数的额外规则。

## 7. 最小可复算例子

向量 $x=(3,4)$ 的 RMS 为

$$
\operatorname{RMS}(x)=\sqrt{\frac{3^2+4^2}{2}}=\frac5{\sqrt2}.
$$

归一化向量为 $(3\sqrt2/5,4\sqrt2/5)$。RMSNorm 不减去均值，因此保留公共偏移方向；它只控制二阶幅度，再乘可学习尺度。

## 8. 如何解释结果

本算法输出所表达的是“仅按均方根缩放激活，保留均值信息并简化层归一化”这一特定数学目标，而不是对数据全部结构的完整描述。解释结果时应直接回到目标函数、距离、概率或集合定义。

- 不能消除整体偏移。

## 9. 计算复杂度

$O(H)$ 每个向量。
