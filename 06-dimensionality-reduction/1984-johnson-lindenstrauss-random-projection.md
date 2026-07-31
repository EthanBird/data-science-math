# Johnson-Lindenstrauss 随机投影（Johnson-Lindenstrauss Random Projection）：数学推导 Tutorial

> 首次提出或经典年份：1984  
> 类别：降维与表示  
> 本章目标：用较低维随机线性映射近似保持有限样本间距离  
> 先修知识：线性代数、特征值分解、拉格朗日乘子

## 1. 问题从哪里来

降维不是简单删除列，而是寻找一个低维坐标，使某种结构尽量不变。不同算法保留的对象可能是方差、类别可分性、相关性、独立性或邻域概率；推导必须从所保留的量出发。

**Johnson-Lindenstrauss 随机投影（Johnson-Lindenstrauss Random Projection）**要解决的具体问题是：用较低维随机线性映射近似保持有限样本间距离。本章不从最终公式开始，而是先定义数据对象和希望保留的关系，再逐步得到公式与算法。

### 1.1 本章将得到什么

- 推导单个向量长度的期望
- 说明最终公式与算法步骤之间的对应关系
- 用一个最小例子检查推导结果

## 2. 数据、符号与假设

有限点集 $\{x_i\}_{i=1}^n\subset\mathbb R^p$，目标维数 $k$，随机矩阵 $R\in\mathbb R^{k\times p}$。

算法输出为：近似保持两两距离的 $k$ 维表示。

为了使上面的数学对象有定义，需要以下前提：

- 中心化、核参数和邻域图均由训练数据确定。
- 特征值并列时只能确定最优子空间，不能唯一确定基向量。

## 3. 建模前的基础数学

### 3.1 低维线性表示

对中心化样本 $x_i-\mu$，线性表示写成

$$
z_i=W^\top(x_i-\mu),\qquad W\in\mathbb R^{p\times k}.
$$

为排除任意缩放，常要求 $W^\top W=I_k$。算法之间的差异在于如何选择 $W$。

### 3.2 广义 Rayleigh 商

许多线性降维目标可写为

$$
R(w)=\frac{w^\top Aw}{w^\top Bw},\qquad B\succ0.
$$

固定 $w^\top Bw=1$，拉格朗日函数为

$$
\mathcal L(w,\lambda)=w^\top Aw-\lambda(w^\top Bw-1).
$$

利用对称矩阵二次型梯度 $\nabla_w(w^\top Aw)=2Aw$，一阶条件给出

$$
Aw=\lambda Bw.
$$

因此 PCA、LDA、CCA 等方法最终会出现普通或广义特征值问题。

## 4. 从定义到算法的完整推导

### A. 单个向量长度的期望

令随机矩阵 $R\in\mathbb R^{k\times p}$ 的元素独立、均值零、方差 $1/k$。对固定 $x$，

$$
\mathbb E\|Rx\|_2^2
=\mathbb E[x^\top R^\top Rx]
=x^\top\mathbb E[R^\top R]x.
$$

第 $a,b$ 个元素为 $\sum_{r=1}^k\mathbb E[R_{ra}R_{rb}]$；当 $a\ne b$ 时为零，当 $a=b$ 时为 $k(1/k)=1$。所以

$$
\mathbb E[R^\top R]=I,
\qquad
\mathbb E\|Rx\|_2^2=\|x\|_2^2.
$$

对次高斯元素使用浓缩不等式可得

$$
\Pr\left(\left|\|Rx\|_2^2-\|x\|_2^2\right|>
\varepsilon\|x\|_2^2\right)
\le2e^{-c k\varepsilon^2}.
$$

对所有 $\binom n2$ 个样本差向量使用 union bound，令失败概率小于给定 $\delta$，得到 $k=O(\varepsilon^{-2}\log(n/\delta))$。

## 5. 把推导压缩成最终公式

前一节给出了公式的来源。本节把最终计算所需的关系集中列出，便于实现时逐项对应。

### 5.1 1. 随机等距缩放

令矩阵元素独立、零均值且方差 $1/k$。

$$
R_{ab}\sim\mathcal N(0,1/k),\qquad z_i=Rx_i.
$$

### 5.2 2. 单向量长度无偏

对固定向量 $v$。

$$
\mathbb E\lVert Rv\rVert_2^2=\lVert v\rVert_2^2.
$$

### 5.3 3. 集中界

长度围绕期望集中。

$$
\Pr\!\left(|\lVert Rv\rVert_2^2-\lVert v\rVert_2^2|>\varepsilon\lVert v\rVert_2^2\right)\le2e^{-c\varepsilon^2k}.
$$

### 5.4 4. 有限点联合界

对所有 $\binom n2$ 个差向量使用并合界。

$$
k=O(\varepsilon^{-2}\log n)\Rightarrow (1-\varepsilon)\lVert x_i-x_j\rVert^2\le\lVert z_i-z_j\rVert^2\le(1+\varepsilon)\lVert x_i-x_j\rVert^2.
$$

## 6. 从公式到算法

**输入：** 有限点集 $\{x_i\}_{i=1}^n\subset\mathbb R^p$，目标维数 $k$，随机矩阵 $R\in\mathbb R^{k\times p}$。
**输出：** 近似保持两两距离的 $k$ 维表示。

1. 由误差容忍度和样本数选 $k$。
2. 用固定随机种子生成投影矩阵。
3. 计算 $Z=XR^\top$。

上面的顺序直接来自前述推导：先计算定义算法状态所需的统计量，再求闭式解、执行递推或更新优化变量，最后按算法定义读取输出。这里没有加入独立于目标函数的额外规则。

## 7. 最小可复算例子

取向量 $x=(1,0,1,0)$、$y=(0,1,0,1)$，原平方距离为 $4$。给定固定投影矩阵

$$
R=\frac1{\sqrt2}\begin{bmatrix}1&1\\1&-1\\-1&1\\-1&-1\end{bmatrix},
$$

计算 $R^\top x=(0,\sqrt2)$、$R^\top y=(0,-\sqrt2)$，投影后平方距离为 $8$。单次小维投影可能有较大失真；JL 引理保证的是维数足够且矩阵随机时，以高概率对有限点集统一控制相对误差。

## 8. 如何解释结果

本算法输出所表达的是“用较低维随机线性映射近似保持有限样本间距离”这一特定数学目标，而不是对数据全部结构的完整描述。解释结果时应直接回到目标函数、距离、概率或集合定义。

- 保证是概率性的。
- 不提供直接可解释的原始特征载荷。

## 9. 计算复杂度

稠密投影为 $O(npk)$；稀疏随机矩阵可降低乘法成本。
