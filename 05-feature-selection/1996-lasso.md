# LASSO（Least Absolute Shrinkage and Selection Operator）：数学推导 Tutorial

> 首次提出或经典年份：1996  
> 类别：特征选择与稀疏化  
> 本章目标：用 $L_1$ 惩罚同时估计和稀疏选择线性特征  
> 先修知识：相关性、互信息、凸优化、次梯度

## 1. 问题从哪里来

特征选择要回答两个不同问题：一个变量是否包含目标信息，以及这部分信息是否已经被其他变量重复表达。过滤法、稀疏正则和错误发现率控制，分别从统计评分、优化和假设检验三个角度处理这个问题。

**LASSO（Least Absolute Shrinkage and Selection Operator）**要解决的具体问题是：用 $L_1$ 惩罚同时估计和稀疏选择线性特征。本章不从最终公式开始，而是先定义数据对象和希望保留的关系，再逐步得到公式与算法。

### 1.1 本章将得到什么

- 推导从坐标子问题推导软阈值更新
- 推导KKT 条件与零系数判据
- 说明最终公式与算法步骤之间的对应关系
- 用一个最小例子检查推导结果

## 2. 数据、符号与假设

设计矩阵 $X\in\mathbb R^{n\times p}$、响应 $y\in\mathbb R^n$，列通常已中心化和缩放。

算法输出为：稀疏系数向量与被选特征集合。

为了使上面的数学对象有定义，需要以下前提：

- 评分或正则化只使用训练数据。
- 变量相关性会影响单变量评分和稀疏解的唯一性。

## 3. 建模前的基础数学

### 3.1 子集选择的组合形式

令 $z_j\in\{0,1\}$ 表示是否选择第 $j$ 个特征。一般形式为

$$
\min_{z\in\{0,1\}^p}\mathcal R(z)+\lambda\sum_{j=1}^p z_j.
$$

若直接枚举，需要比较 $2^p$ 个子集。稀疏正则把离散变量换成连续系数，过滤法构造可快速计算的评分，稳定性与 Knockoff 方法则把选择问题转为统计错误控制。

### 3.2 惩罚形式与约束形式

在适当条件下，

$$
\min_\beta L(\beta)+\lambda\Omega(\beta)
$$

对应于

$$
\min_\beta L(\beta)\quad\text{s.t.}\quad\Omega(\beta)\le t.
$$

$\lambda$ 是约束的拉格朗日乘子，表示增加一单位复杂度需要付出的边际代价。

## 4. 从定义到算法的完整推导

### A. 从坐标子问题推导软阈值更新

固定除第 $j$ 个系数外的所有坐标，定义不含第 $j$ 个变量的部分残差

$$
r_{-j}=y-\sum_{k\ne j}X_k\beta_k.
$$

此时目标中依赖 $\beta_j$ 的部分是

$$
\psi(\beta_j)=\frac1{2n}\|r_{-j}-X_j\beta_j\|_2^2+\lambda|\beta_j|.
$$

展开平方项：

$$
\begin{aligned}
\|r_{-j}-X_j\beta_j\|_2^2
&=(r_{-j}-X_j\beta_j)^\top(r_{-j}-X_j\beta_j)\\
&=r_{-j}^\top r_{-j}-2\beta_jX_j^\top r_{-j}+\beta_j^2X_j^\top X_j.
\end{aligned}
$$

第一项与 $\beta_j$ 无关，可从最小化中删除。记

$$
a_j=\frac1nX_j^\top r_{-j},\qquad c_j=\frac1nX_j^\top X_j>0,
$$

则等价子问题为

$$
\min_b\ \frac12c_jb^2-a_jb+\lambda|b|.
$$

因为绝对值在零点不可微，必须分三种情形。

**情形 1：$b>0$。** 此时 $|b|=b$，普通导数为

$$
\psi'(b)=c_jb-a_j+\lambda.
$$

令其为零：

$$
b=\frac{a_j-\lambda}{c_j}.
$$

该解与假设 $b>0$ 相容当且仅当 $a_j>\lambda$。

**情形 2：$b<0$。** 此时 $|b|=-b$，

$$
\psi'(b)=c_jb-a_j-\lambda=0,
$$

所以

$$
b=\frac{a_j+\lambda}{c_j},
$$

它为负当且仅当 $a_j<-\lambda$。

**情形 3：$b=0$。** 次梯度为

$$
\partial\psi(0)=-a_j+\lambda[-1,1].
$$

由次梯度最优性条件，$0\in\partial\psi(0)$ 等价于

$$
|a_j|\le\lambda.
$$

合并三种情形：

$$
\beta_j\leftarrow\frac{S(a_j,\lambda)}{c_j},
$$

其中

$$
S(a,\lambda)=
\begin{cases}
a-\lambda,&a>\lambda,\\
0,&|a|\le\lambda,\\
a+\lambda,&a<-\lambda.
\end{cases}
$$

再把分段式写成紧凑形式：

$$
S(a,\lambda)=\operatorname{sign}(a)(|a|-\lambda)_+.
$$

这不是经验规则，而是该一维凸子问题的精确闭式解。

### B. KKT 条件与零系数判据

光滑部分记为

$$
g(\beta)=\frac1{2n}\|y-X\beta\|_2^2.
$$

微分为

$$
\begin{aligned}
dg
&=\frac1n(y-X\beta)^\top(-X\,d\beta)\\
&=-\frac1n(X^\top(y-X\beta))^\top d\beta,
\end{aligned}
$$

所以

$$
\nabla g(\beta)=-\frac1nX^\top(y-X\beta).
$$

凸目标的最优性条件是

$$
0\in-\frac1nX^\top(y-X\hat\beta)+\lambda\partial\|\hat\beta\|_1.
$$

逐坐标写成

$$
\frac1nX_j^\top(y-X\hat\beta)=\lambda s_j,
$$

其中 $s_j=\operatorname{sign}(\hat\beta_j)$ 当 $\hat\beta_j\ne0$，而 $s_j\in[-1,1]$ 当 $\hat\beta_j=0$。故零系数必须且只需满足

$$
\left|\frac1nX_j^\top(y-X\hat\beta)\right|\le\lambda.
$$

## 5. 把推导压缩成最终公式

前一节给出了公式的来源。本节把最终计算所需的关系集中列出，便于实现时逐项对应。

### 5.1 1. 稀疏目标

平方误差与 $L_1$ 范数权衡。

$$
\hat\beta=\arg\min_\beta\frac1{2n}\lVert y-X\beta\rVert_2^2+\lambda\lVert\beta\rVert_1.
$$

### 5.2 2. KKT 条件

令残差 $r=y-X\hat\beta$。

$$
\frac1nX_j^\top r=\lambda s_j,\quad s_j=\begin{cases}\operatorname{sign}(\hat\beta_j),&\hat\beta_j\ne0,\\ \left[-1,1\right],&\hat\beta_j=0.\end{cases}
$$

### 5.3 3. 零系数判据

若与残差的相关小于阈值，最优系数为零。

$$
\hat\beta_j=0\Rightarrow \left|\frac1nX_j^\top r\right|\le\lambda.
$$

### 5.4 4. 坐标更新

对标准化列逐坐标最小化。

$$
\beta_j\leftarrow\frac{S\!\left(n^{-1}X_j^\top r_{-j},\lambda\right)}{n^{-1}\lVert X_j\rVert_2^2},\qquad S(a,\lambda)=\operatorname{sign}(a)(|a|-\lambda)_+.
$$

## 6. 从公式到算法

**输入：** 设计矩阵 $X\in\mathbb R^{n\times p}$、响应 $y\in\mathbb R^n$，列通常已中心化和缩放。
**输出：** 稀疏系数向量与被选特征集合。

1. 中心化响应并缩放特征。
2. 选择正则路径。
3. 用坐标下降或近端梯度求解。
4. 按 KKT 条件检查最优性。

上面的顺序直接来自前述推导：先计算定义算法状态所需的统计量，再求闭式解、执行递推或更新优化变量，最后按算法定义读取输出。这里没有加入独立于目标函数的额外规则。

## 7. 最小可复算例子

单变量标准化问题

$$
\min_\beta\frac12(3-\beta)^2+|\beta|.
$$

在 $\beta>0$ 区域，一阶条件为 $\beta-3+1=0$，得 $\beta=2$；在 $\beta<0$ 区域候选不满足符号；$\beta=0$ 的次梯度区间不含零。故解为 $2$，与软阈值 $S_1(3)=2$ 一致。

## 8. 如何解释结果

本算法输出所表达的是“用 $L_1$ 惩罚同时估计和稀疏选择线性特征”这一特定数学目标，而不是对数据全部结构的完整描述。解释结果时应直接回到目标函数、距离、概率或集合定义。

- 高度相关特征之间可能任意选择其一。
- 选择结果依赖 $\lambda$。

## 9. 计算复杂度

每次坐标扫描约 $O(np)$，稀疏设计可更低。
