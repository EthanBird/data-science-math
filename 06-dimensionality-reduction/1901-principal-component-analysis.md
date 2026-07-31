# 主成分分析（Principal Component Analysis）：数学推导 Tutorial

> 首次提出或经典年份：1901  
> 类别：降维与表示  
> 本章目标：寻找保留最大方差的正交线性子空间  
> 先修知识：线性代数、特征值分解、拉格朗日乘子

## 1. 问题从哪里来

降维不是简单删除列，而是寻找一个低维坐标，使某种结构尽量不变。不同算法保留的对象可能是方差、类别可分性、相关性、独立性或邻域概率；推导必须从所保留的量出发。

**主成分分析（Principal Component Analysis）**要解决的具体问题是：寻找保留最大方差的正交线性子空间。本章不从最终公式开始，而是先定义数据对象和希望保留的关系，再逐步得到公式与算法。

### 1.1 本章将得到什么

- 推导从“投影后方差最大”逐行推出特征值问题
- 推导从重构误差逐行推出同一解
- 说明最终公式与算法步骤之间的对应关系
- 用一个最小例子检查推导结果

## 2. 数据、符号与假设

样本矩阵 $X\in\mathbb R^{n\times p}$，中心化后 $X_c=X-\mathbf1\mu^\top$，协方差矩阵 $S=n^{-1}X_c^\top X_c$。

算法输出为：主方向 $W_k$、主成分得分 $Z$ 与解释方差 $\lambda_j$。

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

### A. 从“投影后方差最大”逐行推出特征值问题

令中心化样本为 $x_i-\mu$，选择单位方向 $w\in\mathbb R^p$。第 $i$ 个样本的标量投影定义为

$$
z_i=w^\top(x_i-\mu).
$$

经验均值为

$$
\bar z=\frac1n\sum_{i=1}^n w^\top(x_i-\mu)
=w^\top\left(\frac1n\sum_{i=1}^n(x_i-\mu)\right)
=w^\top 0=0.
$$

第一处等号使用经验均值定义；第二处使用内积对有限求和的线性性；第三处使用 $\mu=n^{-1}\sum_i x_i$。因此投影方差为

$$
\begin{aligned}
\operatorname{Var}(z)
&=\frac1n\sum_{i=1}^n(z_i-\bar z)^2 \\
&=\frac1n\sum_{i=1}^n\left[w^\top(x_i-\mu)\right]^2 \\
&=\frac1n\sum_{i=1}^n w^\top(x_i-\mu)(x_i-\mu)^\top w \\
&=w^\top\left[\frac1n\sum_{i=1}^n(x_i-\mu)(x_i-\mu)^\top\right]w \\
&=w^\top S w.
\end{aligned}
$$

其中第三行使用标量恒等式 $(w^\top a)^2=w^\top aa^\top w$，第四行把与 $i$ 无关的 $w^\top$ 和 $w$ 移到求和号外，第五行代入协方差矩阵定义。

若不限制 $w$，把 $w$ 换成 $cw$ 会使方差乘以 $c^2$，目标无上界。因此加入 $w^\top w=1$。构造拉格朗日函数

$$
\mathcal L(w,\lambda)=w^\top Sw-\lambda(w^\top w-1).
$$

用微分法计算：

$$
\begin{aligned}
d(w^\top Sw)
&=(dw)^\top Sw+w^\top S\,dw \\
&=(dw)^\top Sw+(dw)^\top S^\top w \\
&=2(dw)^\top Sw,
\end{aligned}
$$

最后一步使用 $S=S^\top$。同理 $d(w^\top w)=2(dw)^\top w$，故

$$
d\mathcal L=2(dw)^\top(Sw-\lambda w).
$$

因为 $dw$ 可任意取值，只有

$$
Sw-\lambda w=0
$$

才能使一阶微分恒为零，即 $Sw=\lambda w$。左乘 $w^\top$ 并使用 $w^\top w=1$ 得

$$
w^\top Sw=\lambda,
$$

所以目标值恰为对应特征值。根据 Rayleigh–Ritz 定理，应选最大特征值的特征向量。

### B. 从重构误差逐行推出同一解

令 $W\in\mathbb R^{p\times k}$ 且 $W^\top W=I_k$，正交投影为 $P=WW^\top$。重构误差为

$$
\begin{aligned}
\|X_c-X_cP\|_F^2
&=\operatorname{tr}\left[(X_c-X_cP)^\top(X_c-X_cP)\right]\\
&=\operatorname{tr}\left[X_c^\top X_c-X_c^\top X_cP-PX_c^\top X_c+PX_c^\top X_cP\right]\\
&=\operatorname{tr}(X_c^\top X_c)-2\operatorname{tr}(PX_c^\top X_c)+\operatorname{tr}(PX_c^\top X_cP).
\end{aligned}
$$

由于 $P^2=P$ 且迹满足循环不变性，

$$
\operatorname{tr}(PX_c^\top X_cP)=\operatorname{tr}(P^2X_c^\top X_c)=\operatorname{tr}(PX_c^\top X_c).
$$

因此

$$
\|X_c-X_cP\|_F^2=\operatorname{tr}(X_c^\top X_c)-\operatorname{tr}(W^\top X_c^\top X_cW).
$$

第一项与 $W$ 无关，所以最小化误差等价于最大化 $\operatorname{tr}(W^\top SW)$。谱定理给出最优列空间由最大的 $k$ 个特征向量张成。

## 5. 把推导压缩成最终公式

前一节给出了公式的来源。本节把最终计算所需的关系集中列出，便于实现时逐项对应。

### 5.1 1. 单方向方差

单位方向 $w$ 上的投影为 $z=X_cw$，其经验方差为。

$$
\operatorname{Var}(z)=\frac1n\lVert X_cw\rVert_2^2=w^\top Sw.
$$

### 5.2 2. 约束优化

若不约束长度，方差可任意放大；因此令 $\lVert w\rVert_2=1$。

$$
\max_w\ w^\top Sw\quad\text{s.t.}\quad w^\top w=1.
$$

### 5.3 3. 拉格朗日条件

构造 $L(w,\lambda)=w^\top Sw-\lambda(w^\top w-1)$。

$$
\nabla_wL=2Sw-2\lambda w=0\Rightarrow Sw=\lambda w.
$$

### 5.4 4. 多维子空间

取最大 $k$ 个特征值对应的正交特征向量。

$$
W_k=[w_1,\ldots,w_k],\qquad Z=X_cW_k.
$$

### 5.5 5. 最小重构误差

同一解也最小化正交投影的平方重构误差。

$$
\min_{W^\top W=I_k}\lVert X_c-X_cWW^\top\rVert_F^2=\sum_{j>k}\lambda_j.
$$

## 6. 从公式到算法

**输入：** 样本矩阵 $X\in\mathbb R^{n\times p}$，中心化后 $X_c=X-\mathbf1\mu^\top$，协方差矩阵 $S=n^{-1}X_c^\top X_c$。
**输出：** 主方向 $W_k$、主成分得分 $Z$ 与解释方差 $\lambda_j$。

1. 按训练集均值中心化。
2. 计算协方差矩阵或直接做 SVD。
3. 按特征值降序选择前 $k$ 个方向。
4. 投影得到低维坐标。

上面的顺序直接来自前述推导：先计算定义算法状态所需的统计量，再求闭式解、执行递推或更新优化变量，最后按算法定义读取输出。这里没有加入独立于目标函数的额外规则。

## 7. 最小可复算例子

中心化样本为 $x_1=(-1,-1)$、$x_2=(1,1)$。协方差矩阵

$$
S=\frac12\left[x_1x_1^\top+x_2x_2^\top\right]
=\begin{bmatrix}1&1\\1&1\end{bmatrix}.
$$

特征值为 $2,0$，最大特征向量可取 $w=(1,1)^\top/\sqrt2$。投影得 $(-\sqrt2,\sqrt2)$，方差为 $2$；正交方向 $(1,-1)^\top/\sqrt2$ 的投影恒为零。

## 8. 如何解释结果

本算法输出所表达的是“寻找保留最大方差的正交线性子空间”这一特定数学目标，而不是对数据全部结构的完整描述。解释结果时应直接回到目标函数、距离、概率或集合定义。

- 只描述线性子空间。
- 方差大不等价于任务相关。

## 9. 计算复杂度

协方差特征分解约为 $O(np^2+p^3)$；截断 SVD 可降低成本。
