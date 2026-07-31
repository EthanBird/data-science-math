# 稳健主成分分析（Robust PCA / Principal Component Pursuit）：数学推导 Tutorial

> 首次提出或经典年份：2011  
> 类别：矩阵分解、补全与最优传输  
> 本章目标：把观测矩阵分解为低秩主体与稀疏异常  
> 先修知识：矩阵微分、SVD、凸优化与概率单纯形

## 1. 问题从哪里来

矩阵分解把大量观测解释为少数潜在成分。低秩、非负、稀疏或边际约束规定了这些成分的含义，求解通常来自交替最小化、近端算子或矩阵缩放。

**稳健主成分分析（Robust PCA / Principal Component Pursuit）**要解决的具体问题是：把观测矩阵分解为低秩主体与稀疏异常。本章不从最终公式开始，而是先定义数据对象和希望保留的关系，再逐步得到公式与算法。

### 1.1 本章将得到什么

- 推导低秩加稀疏分解
- 说明最终公式与算法步骤之间的对应关系
- 用一个最小例子检查推导结果

## 2. 数据、符号与假设

矩阵 $M=L+S$，其中 $L$ 低秩、$S$ 元素级稀疏。

算法输出为：低秩背景 $L$ 与稀疏异常 $S$。

为了使上面的数学对象有定义，需要以下前提：

- 观测集合、秩或正则化参数已给定。
- 非凸分解的结果可能依赖初始化。

## 3. 建模前的基础数学

### 3.1 低复杂度重构

给定 $X\in\mathbb R^{n\times p}$，一般分解问题写成

$$
\min_{A,B}\frac12\|X-AB\|_F^2+\mathcal R(A,B).
$$

展开 Frobenius 范数：

$$
\|X-AB\|_F^2=\sum_{i=1}^n\sum_{j=1}^p\left(X_{ij}-\sum_{r=1}^kA_{ir}B_{rj}\right)^2.
$$

固定 $B$ 后，光滑部分对 $A$ 的梯度为

$$
\nabla_A\frac12\|X-AB\|_F^2=(AB-X)B^\top.
$$

若没有额外约束并且 $BB^\top$ 可逆，正规方程给出

$$
A=XB^\top(BB^\top)^{-1}.
$$

非负、稀疏或边际约束会把闭式解改为投影、近端或乘法更新。

## 4. 从定义到算法的完整推导

### A. 低秩加稀疏分解

假设

$$
X=L+S,
$$

其中 $L$ 低秩、$S$ 元素稀疏。直接最小化

$$
\operatorname{rank}(L)+\lambda\|S\|_0
$$

不可解性强。用各自的凸包代理得到 Principal Component Pursuit：

$$
\min_{L,S}\ \|L\|_*+\lambda\|S\|_1
\quad\text{s.t.}\quad X=L+S.
$$

增广拉格朗日交替更新时，$L$ 子问题是奇异值软阈值，$S$ 子问题是逐元素软阈值：

$$
L^{k+1}=\operatorname{SVT}_{1/\mu}(X-S^k+Y^k/\mu),
$$

$$
S^{k+1}=S_{\lambda/\mu}(X-L^{k+1}+Y^k/\mu).
$$

随后对偶更新 $Y^{k+1}=Y^k+\mu(X-L^{k+1}-S^{k+1})$。

## 5. 把推导压缩成最终公式

前一节给出了公式的来源。本节把最终计算所需的关系集中列出，便于实现时逐项对应。

### 5.1 1. 理想组合问题

$$
\min_{L,S}\operatorname{rank}(L)+\gamma\lVert S\rVert_0\quad\text{s.t.}\quad M=L+S.
$$

### 5.2 2. 凸松弛

用核范数和 $L_1$ 范数替代秩与零范数。

$$
\min_{L,S}\lVert L\rVert_*+\lambda\lVert S\rVert_1\quad\text{s.t.}\quad M=L+S.
$$

### 5.3 3. 增广拉格朗日

$$
\mathcal L_\mu=\lVert L\rVert_*+\lambda\lVert S\rVert_1+\langle Y,M-L-S\rangle+\frac\mu2\lVert M-L-S\rVert_F^2.
$$

### 5.4 4. 交替近端

分别得到奇异值阈值和元素软阈值。

$$
L\leftarrow\operatorname{SVT}_{1/\mu}(M-S+Y/\mu),\qquad S\leftarrow S_{\lambda/\mu}(M-L+Y/\mu).
$$

## 6. 从公式到算法

**输入：** 矩阵 $M=L+S$，其中 $L$ 低秩、$S$ 元素级稀疏。
**输出：** 低秩背景 $L$ 与稀疏异常 $S$。

1. 选择 $\lambda$ 并初始化。
2. 交替更新低秩矩阵与稀疏矩阵。
3. 更新对偶变量。
4. 原始残差足够小时停止。

上面的顺序直接来自前述推导：先计算定义算法状态所需的统计量，再求闭式解、执行递推或更新优化变量，最后按算法定义读取输出。这里没有加入独立于目标函数的额外规则。

## 7. 最小可复算例子

观测矩阵

$$
M=\begin{bmatrix}1&1\\1&10\end{bmatrix}
$$

可看作低秩主体

$$
L=\begin{bmatrix}1&1\\1&1\end{bmatrix}
$$

加稀疏异常 $S_{22}=9$。Principal Component Pursuit 求

$$
\min_{L,S}\|L\|_*+\lambda\|S\|_1\quad\text{s.t. }M=L+S,
$$

用核范数鼓励低秩，用 $L_1$ 鼓励少量异常位置。

## 8. 如何解释结果

本算法输出所表达的是“把观测矩阵分解为低秩主体与稀疏异常”这一特定数学目标，而不是对数据全部结构的完整描述。解释结果时应直接回到目标函数、距离、概率或集合定义。

- 异常必须足够稀疏且低秩部分不能过于集中在少数坐标。

## 9. 计算复杂度

每轮主要为 SVD。
