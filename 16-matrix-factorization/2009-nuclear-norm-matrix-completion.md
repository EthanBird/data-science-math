# 核范数矩阵补全（Nuclear-Norm Matrix Completion）：数学推导 Tutorial

> 首次提出或经典年份：2009  
> 类别：矩阵分解、补全与最优传输  
> 本章目标：用凸低秩代理从部分观测恢复矩阵  
> 先修知识：矩阵微分、SVD、凸优化与概率单纯形

## 1. 问题从哪里来

矩阵分解把大量观测解释为少数潜在成分。低秩、非负、稀疏或边际约束规定了这些成分的含义，求解通常来自交替最小化、近端算子或矩阵缩放。

**核范数矩阵补全（Nuclear-Norm Matrix Completion）**要解决的具体问题是：用凸低秩代理从部分观测恢复矩阵。本章不从最终公式开始，而是先定义数据对象和希望保留的关系，再逐步得到公式与算法。

### 1.1 本章将得到什么

- 推导从秩最小化到核范数松弛
- 说明最终公式与算法步骤之间的对应关系
- 用一个最小例子检查推导结果

## 2. 数据、符号与假设

未知矩阵 $M\in\mathbb R^{n\times p}$，观测索引集合 $\Omega$，投影算子 $P_\Omega(X)_{ij}=X_{ij}$ 若 $(i,j)\in\Omega$，否则为零。

算法输出为：低秩补全矩阵。

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

### A. 从秩最小化到核范数松弛

理想问题

$$
\min_Z\operatorname{rank}(Z)
\quad\text{s.t.}\quad\mathcal P_\Omega(Z)=\mathcal P_\Omega(X)
$$

是非凸组合问题。矩阵秩等于非零奇异值个数，即奇异值向量的 $L_0$“范数”。在谱范数单位球上，秩函数的凸包是核范数

$$
\|Z\|_*=\sum_i\sigma_i(Z).
$$

因此使用凸松弛

$$
\min_Z\|Z\|_*
\quad\text{s.t.}\quad\mathcal P_\Omega(Z)=\mathcal P_\Omega(X).
$$

带噪情形转为观测平方损失加核范数惩罚；其近端步骤是奇异值软阈值。

## 5. 把推导压缩成最终公式

前一节给出了公式的来源。本节把最终计算所需的关系集中列出，便于实现时逐项对应。

### 5.1 1. 理想低秩目标

最直接问题是最小秩补全，但秩最小化是组合问题。

$$
\min_X\operatorname{rank}(X)\quad\text{s.t.}\quad P_\Omega(X)=P_\Omega(M).
$$

### 5.2 2. 凸包代理

核范数是奇异值之和，是谱范数单位球上秩函数的凸包。

$$
\lVert X\rVert_*=\sum_j\sigma_j(X).
$$

### 5.3 3. 凸松弛

$$
\min_X\lVert X\rVert_*\quad\text{s.t.}\quad P_\Omega(X)=P_\Omega(M).
$$

### 5.4 4. 含噪形式

$$
\min_X\frac12\lVert P_\Omega(X-M)\rVert_F^2+\lambda\lVert X\rVert_*.
$$

### 5.5 5. 近端算子

核范数近端是奇异值软阈值。

$$
X=U\operatorname{diag}(\sigma)V^\top\Rightarrow\operatorname{prox}_{\tau\lVert\cdot\rVert_*}(X)=U\operatorname{diag}((\sigma-\tau)_+)V^\top.
$$

## 6. 从公式到算法

**输入：** 未知矩阵 $M\in\mathbb R^{n\times p}$，观测索引集合 $\Omega$，投影算子 $P_\Omega(X)_{ij}=X_{ij}$ 若 $(i,j)\in\Omega$，否则为零。
**输出：** 低秩补全矩阵。

1. 构造观测投影。
2. 选择约束或含噪目标。
3. 用近端梯度/ADMM 交替做观测梯度和奇异值阈值。
4. 检查观测残差与目标变化。

上面的顺序直接来自前述推导：先计算定义算法状态所需的统计量，再求闭式解、执行递推或更新优化变量，最后按算法定义读取输出。这里没有加入独立于目标函数的额外规则。

## 7. 最小可复算例子

秩一矩阵

$$
X^\star=\begin{bmatrix}1&2\\2&4\end{bmatrix}
$$

若只观测三个位置，可求

$$
\min_Z\frac12\|\mathcal P_\Omega(Z-X^\star)\|_F^2+\lambda\|Z\|_*.
$$

核范数是奇异值之和，倾向少数非零奇异值。近端梯度先沿观测误差更新，再对奇异值做软阈值，缺失位置由低秩结构间接确定。

## 8. 如何解释结果

本算法输出所表达的是“用凸低秩代理从部分观测恢复矩阵”这一特定数学目标，而不是对数据全部结构的完整描述。解释结果时应直接回到目标函数、距离、概率或集合定义。

- 需要低秩和观测充分分散等条件。
- 非随机缺失可能不可识别。

## 9. 计算复杂度

主要成本为重复截断 SVD。
