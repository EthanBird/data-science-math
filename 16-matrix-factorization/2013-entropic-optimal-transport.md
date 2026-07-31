# 熵正则最优传输（Entropic Optimal Transport）：数学推导 Tutorial

> 首次提出或经典年份：2013  
> 类别：矩阵分解、补全与最优传输  
> 本章目标：用熵正则化快速求两个离散分布间的软匹配  
> 先修知识：矩阵微分、SVD、凸优化与概率单纯形

## 1. 问题从哪里来

矩阵分解把大量观测解释为少数潜在成分。低秩、非负、稀疏或边际约束规定了这些成分的含义，求解通常来自交替最小化、近端算子或矩阵缩放。

**熵正则最优传输（Entropic Optimal Transport）**要解决的具体问题是：用熵正则化快速求两个离散分布间的软匹配。本章不从最终公式开始，而是先定义数据对象和希望保留的关系，再逐步得到公式与算法。

### 1.1 本章将得到什么

- 推导KKT 条件推出 Gibbs 缩放形式
- 说明最终公式与算法步骤之间的对应关系
- 用一个最小例子检查推导结果

## 2. 数据、符号与假设

边际分布 $a\in\Delta_n,b\in\Delta_m$，成本矩阵 $C\in\mathbb R^{n\times m}$，耦合集合 $\Pi(a,b)$。

算法输出为：平滑最优传输计划与正则化传输成本。

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

### A. KKT 条件推出 Gibbs 缩放形式

耦合集

$$
\Pi(a,b)=\{P\ge0:P\mathbf1=a,\ P^\top\mathbf1=b\}.
$$

目标为

$$
\langle C,P\rangle+
\varepsilon\sum_{ij}P_{ij}(\log P_{ij}-1).
$$

给行列约束乘子 $f_i,g_j$，拉格朗日函数中的单元项为

$$
C_{ij}P_{ij}+\varepsilon P_{ij}(\log P_{ij}-1)-f_iP_{ij}-g_jP_{ij}.
$$

利用

$$
\frac d{dp}[p(\log p-1)]=\log p,
$$

驻点条件是

$$
C_{ij}+\varepsilon\log P_{ij}-f_i-g_j=0.
$$

解对数：

$$
P_{ij}=
\exp(f_i/\varepsilon)
\exp(-C_{ij}/\varepsilon)
\exp(g_j/\varepsilon).
$$

令 $u_i=e^{f_i/\varepsilon}$、$K_{ij}=e^{-C_{ij}/\varepsilon}$、$v_j=e^{g_j/\varepsilon}$，得到

$$
P=\operatorname{diag}(u)K\operatorname{diag}(v).
$$

边际约束再推出 Sinkhorn 更新。

## 5. 把推导压缩成最终公式

前一节给出了公式的来源。本节把最终计算所需的关系集中列出，便于实现时逐项对应。

### 5.1 1. 正则目标

$$
\min_{P\in\Pi(a,b)}\langle C,P\rangle+\varepsilon\sum_{i,j}P_{ij}(\log P_{ij}-1).
$$

### 5.2 2. 拉格朗日驻点

对 $P_{ij}$ 求导。

$$
C_{ij}+\varepsilon\log P_{ij}-f_i-g_j=0.
$$

### 5.3 3. 缩放形式

$$
P_{ij}=u_iK_{ij}v_j,\qquad K_{ij}=e^{-C_{ij}/\varepsilon}.
$$

### 5.4 4. 边际方程

$$
u=a\oslash(Kv),\qquad v=b\oslash(K^\top u).
$$

## 6. 从公式到算法

**输入：** 边际分布 $a\in\Delta_n,b\in\Delta_m$，成本矩阵 $C\in\mathbb R^{n\times m}$，耦合集合 $\Pi(a,b)$。
**输出：** 平滑最优传输计划与正则化传输成本。

1. 计算 Gibbs 核 $K$。
2. 初始化正缩放向量。
3. 交替执行行列边际缩放。
4. 检查边际残差并输出耦合。

上面的顺序直接来自前述推导：先计算定义算法状态所需的统计量，再求闭式解、执行递推或更新优化变量，最后按算法定义读取输出。这里没有加入独立于目标函数的额外规则。

## 7. 最小可复算例子

两个离散分布边际均为 $(1/2,1/2)$，代价矩阵

$$
C=\begin{bmatrix}0&1\\1&0\end{bmatrix}.
$$

无熵正则时最优耦合把质量放在对角线。加入 $\varepsilon\sum_{ij}P_{ij}(\log P_{ij}-1)$ 后，核矩阵 $K=e^{-C/\varepsilon}$ 全为正，Sinkhorn 缩放得到平滑耦合；$\varepsilon$ 越小，质量越集中到低代价对角线。

## 8. 如何解释结果

本算法输出所表达的是“用熵正则化快速求两个离散分布间的软匹配”这一特定数学目标，而不是对数据全部结构的完整描述。解释结果时应直接回到目标函数、距离、概率或集合定义。

- $\varepsilon$ 大会产生偏平滑耦合，小则数值下溢，常需对数域实现。

## 9. 计算复杂度

稠密每轮 $O(nm)$。
