# 期望最大化算法（Expectation-Maximization Algorithm）：数学推导 Tutorial

> 首次提出或经典年份：1977  
> 类别：缺失数据  
> 本章目标：在隐变量或不完整数据下迭代最大化观测似然  
> 先修知识：经验风险、条件期望、矩阵范数与低秩表示

## 1. 问题从哪里来

缺失值处理的难点不是把空格填上数字，而是区分“已经观测到的约束”和“对未观测量的估计”。不同填补方法对应不同损失函数、相似性假设或低秩模型，因此必须从估计目标本身推导。

**期望最大化算法（Expectation-Maximization Algorithm）**要解决的具体问题是：在隐变量或不完整数据下迭代最大化观测似然。本章不从最终公式开始，而是先定义数据对象和希望保留的关系，再逐步得到公式与算法。

### 1.1 本章将得到什么

- 推导观测似然为什么难以直接最大化
- 推导用任意分布 $q$ 构造下界
- 推导下界间隙严格等于 KL 散度
- 推导E 步、M 步与单调性
- 说明最终公式与算法步骤之间的对应关系
- 用一个最小例子检查推导结果

## 2. 数据、符号与假设

观测数据 $X$、隐变量 $Z$、参数 $\theta$，联合模型为 $p(X,Z\mid\theta)$，观测似然为 $p(X\mid\theta)=\sum_Zp(X,Z\mid\theta)$。

算法输出为：局部极值处的参数估计和隐变量后验。

为了使上面的数学对象有定义，需要以下前提：

- 缺失掩码已知，已观测条目不被无意改写。
- 填补值的统计解释依赖缺失机制与模型假设。

## 3. 建模前的基础数学

### 3.1 观测掩码

令完整矩阵为 $X^\star\in\mathbb R^{n\times p}$，观测掩码为

$$
M_{ij}=\begin{cases}1,&X^\star_{ij}\text{ 已观测},\\0,&X^\star_{ij}\text{ 缺失}.
\end{cases}
$$

定义观测投影

$$
[\mathcal P_\Omega(X)]_{ij}=M_{ij}X_{ij}.
$$

任何填补结果 $Z$ 至少应满足

$$
\mathcal P_\Omega(Z)=\mathcal P_\Omega(X^\star),
$$

即不能在没有说明的情况下改写已观测位置。

### 3.2 常数填补的统一风险形式

若用常数 $a$ 填补一列，可写成

$$
a^\star=\arg\min_a\sum_{i\in\Omega}\ell(x_i,a).
$$

平方损失、绝对损失和零一损失分别导出均值、中位数和众数。

## 4. 从定义到算法的完整推导

### A. 观测似然为什么难以直接最大化

由全概率公式，离散隐变量情形有

$$
p(X\mid\theta)=\sum_Zp(X,Z\mid\theta).
$$

取对数得到

$$
\ell(\theta)=\log\sum_Zp(X,Z\mid\theta).
$$

困难不在求和本身，而在“对数包住求和”：通常

$$
\log\sum_Z a_Z\ne\sum_Z\log a_Z,
$$

因此无法把不同隐状态的贡献拆开独立优化。

### B. 用任意分布 $q$ 构造下界

取任意满足 $q(Z)>0$ 且 $\sum_Zq(Z)=1$ 的分布。先乘除同一个正数 $q(Z)$：

$$
\begin{aligned}
\ell(\theta)
&=\log\sum_Zp(X,Z\mid\theta)\\
&=\log\sum_Zq(Z)\frac{p(X,Z\mid\theta)}{q(Z)}\\
&=\log\mathbb E_q\left[\frac{p(X,Z\mid\theta)}{q(Z)}\right].
\end{aligned}
$$

第一到第二行只是恒等变形；第二到第三行使用离散期望定义。由于 $\log$ 是凹函数，Jensen 不等式给出

$$
\ell(\theta)\ge\mathbb E_q\left[\log\frac{p(X,Z\mid\theta)}{q(Z)}\right].
$$

展开对数商：

$$
\begin{aligned}
\mathcal L(q,\theta)
&=\mathbb E_q[\log p(X,Z\mid\theta)-\log q(Z)]\\
&=\mathbb E_q[\log p(X,Z\mid\theta)]+H(q),
\end{aligned}
$$

其中 $H(q)=-\mathbb E_q\log q(Z)$ 是熵的定义。

### C. 下界间隙严格等于 KL 散度

由 Bayes 公式

$$
p(Z\mid X,\theta)=\frac{p(X,Z\mid\theta)}{p(X\mid\theta)}.
$$

将它代入 KL 散度：

$$
\begin{aligned}
\operatorname{KL}\left(q\|p(Z\mid X,\theta)\right)
&=\mathbb E_q\left[\log\frac{q(Z)}{p(Z\mid X,\theta)}\right]\\
&=\mathbb E_q\left[\log q(Z)-\log p(X,Z\mid\theta)+\log p(X\mid\theta)\right]\\
&=\log p(X\mid\theta)-\mathcal L(q,\theta).
\end{aligned}
$$

最后一步使用 $\log p(X\mid\theta)$ 与 $Z$ 无关，故其期望仍等于自身。于是

$$
\log p(X\mid\theta)=\mathcal L(q,\theta)+\operatorname{KL}\left(q\|p(Z\mid X,\theta)\right).
$$

KL 散度非负，所以 $\mathcal L$ 是下界；等号当且仅当

$$
q(Z)=p(Z\mid X,\theta)
$$

几乎处处成立。这一步直接推出 E 步。

### D. E 步、M 步与单调性

在第 $t$ 轮，E 步令

$$
q^{(t)}(Z)=p(Z\mid X,\theta^{(t)}),
$$

因此下界在当前参数处贴紧：

$$
\mathcal L(q^{(t)},\theta^{(t)})=\ell(\theta^{(t)}).
$$

M 步选择

$$
\theta^{(t+1)}\in\arg\max_\theta\mathcal L(q^{(t)},\theta).
$$

熵 $H(q^{(t)})$ 在 M 步中是常数，所以等价于最大化

$$
Q(\theta\mid\theta^{(t)})=\mathbb E_{q^{(t)}}[\log p(X,Z\mid\theta)].
$$

由“下界不超过似然”和“M 步不降低下界”，有完整不等式链

$$
\begin{aligned}
\ell(\theta^{(t+1)})
&\ge \mathcal L(q^{(t)},\theta^{(t+1)})\\
&\ge \mathcal L(q^{(t)},\theta^{(t)})\\
&=\ell(\theta^{(t)}).
\end{aligned}
$$

这证明 EM 的观测对数似然单调不下降；它没有证明收敛到全局最大值。

## 5. 把推导压缩成最终公式

前一节给出了公式的来源。本节把最终计算所需的关系集中列出，便于实现时逐项对应。

### 5.1 1. 引入任意隐变量分布

对任意 $q(Z)$ 重写对数似然。

$$
\log p(X\mid\theta)=\log\sum_Zq(Z)\frac{p(X,Z\mid\theta)}{q(Z)}.
$$

### 5.2 2. Jensen 下界

对数是凹函数。

$$
\log p(X\mid\theta)\ge\mathcal L(q,\theta)=\mathbb E_q[\log p(X,Z\mid\theta)]+H(q).
$$

### 5.3 3. E 步

固定参数时，下界在后验处分布达到等号。

$$
q^{(t+1)}(Z)=p(Z\mid X,\theta^{(t)}).
$$

### 5.4 4. M 步

固定 $q$ 最大化期望完整数据对数似然。

$$
\theta^{(t+1)}=\arg\max_\theta Q(\theta\mid\theta^{(t)}),\quad Q=\mathbb E_{Z\mid X,\theta^{(t)}}[\log p(X,Z\mid\theta)].
$$

### 5.5 5. 单调性

交替坐标上升保证观测似然不下降。

$$
\log p(X\mid\theta^{(t+1)})\ge\log p(X\mid\theta^{(t)}).
$$

## 6. 从公式到算法

**输入：** 观测数据 $X$、隐变量 $Z$、参数 $\theta$，联合模型为 $p(X,Z\mid\theta)$，观测似然为 $p(X\mid\theta)=\sum_Zp(X,Z\mid\theta)$。
**输出：** 局部极值处的参数估计和隐变量后验。

1. 初始化参数。
2. 计算隐变量后验或其充分统计量。
3. 最大化期望完整数据对数似然。
4. 检查观测似然增量。

上面的顺序直接来自前述推导：先计算定义算法状态所需的统计量，再求闭式解、执行递推或更新优化变量，最后按算法定义读取输出。这里没有加入独立于目标函数的额外规则。

## 7. 最小可复算例子

考虑两个一维高斯成分，当前参数为 $\pi_1=\pi_2=1/2$、$\mu_1=0$、$\mu_2=4$、方差均为 $1$，观测 $x=1$。责任度比为

$$
\frac{\gamma_1}{\gamma_2}
=\frac{\pi_1e^{-(1-0)^2/2}}{\pi_2e^{-(1-4)^2/2}}
=e^{4}.
$$

所以 $\gamma_1=e^4/(1+e^4)\approx0.982$。E 步把硬分配替换成后验权重；M 步再用所有样本的这些权重计算加权均值和方差。

## 8. 如何解释结果

本算法输出所表达的是“在隐变量或不完整数据下迭代最大化观测似然”这一特定数学目标，而不是对数据全部结构的完整描述。解释结果时应直接回到目标函数、距离、概率或集合定义。

- 只保证收敛到驻点，不保证全局最优。
- 不同初始化可能得到不同解。

## 9. 计算复杂度

每轮取决于后验计算和 M 步优化成本。
