# Box-Cox 变换（Box-Cox Transform）：数学推导 Tutorial

> 首次提出或经典年份：1964  
> 类别：缩放、变换与离散化  
> 本章目标：通过参数化幂变换稳定方差并改善正态近似  
> 先修知识：一元函数、仿射变换、导数、经验分布函数

## 1. 问题从哪里来

缩放和变换会直接改变距离、梯度、协方差和正则化的含义。本类方法的核心是构造一个映射，并证明它满足目标端点、单调性、均值方差或分布匹配等要求。

**Box-Cox 变换（Box-Cox Transform）**要解决的具体问题是：通过参数化幂变换稳定方差并改善正态近似。本章不从最终公式开始，而是先定义数据对象和希望保留的关系，再逐步得到公式与算法。

### 1.1 本章将得到什么

- 推导$\lambda=0$ 分支由极限而来
- 推导Jacobian 项如何进入似然
- 说明最终公式与算法步骤之间的对应关系
- 用一个最小例子检查推导结果

## 2. 数据、符号与假设

正值样本 $x_i>0$，变换参数 $\lambda\in\mathbb R$。

算法输出为：变换参数 $\hat\lambda$ 与变换后数据。

为了使上面的数学对象有定义，需要以下前提：

- 所有分母必须非零；幂函数和对数满足各自定义域。
- 训练集估计的端点、均值、方差或分位点在测试阶段保持固定。

## 3. 建模前的基础数学

### 3.1 从端点条件推导一般仿射缩放

设 $T(x)=ax+b$，要求 $u\mapsto\alpha$、$v\mapsto\beta$。于是

$$
au+b=\alpha,\qquad av+b=\beta.
$$

两式相减得到

$$
a(v-u)=\beta-\alpha.
$$

当 $u\ne v$ 时，

$$
a=\frac{\beta-\alpha}{v-u},\qquad b=\alpha-au,
$$

所以

$$
T(x)=\alpha+\frac{x-u}{v-u}(\beta-\alpha).
$$

这一步说明 Min-Max、单位变换等公式不是需要死记的规则，而是两个端点条件唯一确定的仿射函数。

### 3.2 单调变换保留次序

若 $T$ 严格递增，则 $x_i<x_j\Rightarrow T(x_i)<T(x_j)$。因此它改变数值间距但不改变排序；若只单调不减，则可能把不同输入压成并列值。

## 4. 从定义到算法的完整推导

### A. $\lambda=0$ 分支由极限而来

当 $\lambda\ne0$ 时

$$
T_\lambda(x)=\frac{x^\lambda-1}{\lambda}.
$$

写成指数形式 $x^\lambda=e^{\lambda\log x}$。对 $e^u$ 在 $u=0$ 处作 Taylor 展开：

$$
e^{\lambda\log x}=1+\lambda\log x+O(\lambda^2).
$$

所以

$$
\frac{x^\lambda-1}{\lambda}
=\log x+O(\lambda),
$$

令 $\lambda\to0$ 得

$$
\lim_{\lambda\to0}T_\lambda(x)=\log x.
$$

因此对数分支保证参数连续，而不是任意拼接。

### B. Jacobian 项如何进入似然

设 $z_i=T_\lambda(x_i)$ 且假设 $z_i\overset{\text{iid}}\sim\mathcal N(\mu,\sigma^2)$。变量代换公式给出

$$
p_X(x_i)=p_Z(T_\lambda(x_i))\left|\frac{dT_\lambda(x_i)}{dx_i}\right|.
$$

当 $\lambda\ne0$，导数为

$$
\frac{dT_\lambda(x)}{dx}=x^{\lambda-1}.
$$

因此对数 Jacobian 为 $(\lambda-1)\log x_i$。把高斯对数密度和 Jacobian 相加并对 $i$ 求和，得到

$$
\ell=-\frac n2\log(2\pi\sigma^2)
-\frac1{2\sigma^2}\sum_i(z_i-\mu)^2
+(\lambda-1)\sum_i\log x_i.
$$

固定 $\lambda$ 后，对 $\mu$ 和 $\sigma^2$ 求导，分别得到 $\hat\mu=\bar z$ 与 $\hat\sigma^2=n^{-1}\sum_i(z_i-\bar z)^2$；代回即为 profile likelihood。

## 5. 把推导压缩成最终公式

前一节给出了公式的来源。本节把最终计算所需的关系集中列出，便于实现时逐项对应。

### 5.1 1. 连续幂变换族

除以 $\lambda$ 使 $\lambda\to0$ 时连续收敛到对数。

$$
T_\lambda(x)=\begin{cases}(x^\lambda-1)/\lambda,&\lambda\ne0,\\\log x,&\lambda=0.\end{cases}
$$

### 5.2 2. 高斯似然

假设变换后 $z_i=T_\lambda(x_i)$ 服从同方差高斯分布。

$$
\ell(\mu,\sigma^2,\lambda)=-\frac n2\log\sigma^2-\frac1{2\sigma^2}\sum_i(z_i-\mu)^2+(\lambda-1)\sum_i\log x_i+C.
$$

### 5.3 3. 剖面似然

对固定 $\lambda$，$\mu,\sigma^2$ 有闭式解。

$$
\hat\mu_\lambda=\bar z_\lambda,\qquad \hat\sigma_\lambda^2=n^{-1}\sum_i(z_i-\bar z_\lambda)^2.
$$

### 5.4 4. 参数选择

一维最大化剖面似然。

$$
\hat\lambda=\arg\max_\lambda\ell(\hat\mu_\lambda,\hat\sigma_\lambda^2,\lambda).
$$

## 6. 从公式到算法

**输入：** 正值样本 $x_i>0$，变换参数 $\lambda\in\mathbb R$。
**输出：** 变换参数 $\hat\lambda$ 与变换后数据。

1. 确认数据严格为正。
2. 在候选区间计算每个 $\lambda$ 的剖面似然。
3. 用一维优化求最大值。
4. 应用对应幂变换。

上面的顺序直接来自前述推导：先计算定义算法状态所需的统计量，再求闭式解、执行递推或更新优化变量，最后按算法定义读取输出。这里没有加入独立于目标函数的额外规则。

## 7. 最小可复算例子

取 $x=4$。当 $\lambda=1/2$ 时

$$
T_{1/2}(4)=\frac{4^{1/2}-1}{1/2}=2.
$$

当 $\lambda\to0$，利用 $x^\lambda=e^{\lambda\log x}=1+\lambda\log x+o(\lambda)$，有

$$
\lim_{\lambda\to0}\frac{x^\lambda-1}{\lambda}=\log x.
$$

所以分段定义在 $\lambda=0$ 处连续。

## 8. 如何解释结果

本算法输出所表达的是“通过参数化幂变换稳定方差并改善正态近似”这一特定数学目标，而不是对数据全部结构的完整描述。解释结果时应直接回到目标函数、距离、概率或集合定义。

- 非正数据不在定义域。
- 改善正态性不是必然结果。

## 9. 计算复杂度

每次似然评估 $O(n)$；总量取决于一维搜索次数。
