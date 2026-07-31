# Knockoff 过滤器（Knockoff Filter）：数学推导 Tutorial

> 首次提出或经典年份：2015  
> 类别：特征选择与稀疏化  
> 本章目标：在控制错误发现率的前提下进行变量选择  
> 先修知识：相关性、互信息、凸优化、次梯度

## 1. 问题从哪里来

特征选择要回答两个不同问题：一个变量是否包含目标信息，以及这部分信息是否已经被其他变量重复表达。过滤法、稀疏正则和错误发现率控制，分别从统计评分、优化和假设检验三个角度处理这个问题。

**Knockoff 过滤器（Knockoff Filter）**要解决的具体问题是：在控制错误发现率的前提下进行变量选择。本章不从最终公式开始，而是先定义数据对象和希望保留的关系，再逐步得到公式与算法。

### 1.1 本章将得到什么

- 推导固定设计 Knockoff 的 Gram 约束
- 推导反对称统计量
- 推导阈值
- 说明最终公式与算法步骤之间的对应关系
- 用一个最小例子检查推导结果

## 2. 数据、符号与假设

固定设计矩阵 $X\in\mathbb R^{n\times p}$，构造 knockoff 矩阵 $\tilde X$，要求交换原变量与对应 knockoff 后 Gram 结构不变。

算法输出为：带有限样本 FDR 控制的特征集合。

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

### A. 固定设计 Knockoff 的 Gram 约束

希望原变量与替身具有相同二阶几何。令 $\Sigma=X^\top X$，规定

$$
\widetilde X^\top\widetilde X=\Sigma,
\qquad
X^\top\widetilde X=\Sigma-\operatorname{diag}(s).
$$

于是联合 Gram 矩阵为

$$
G=
\begin{bmatrix}
\Sigma&\Sigma-D_s\\
\Sigma-D_s&\Sigma
\end{bmatrix}.
$$

存在实矩阵 $[X,\widetilde X]$ 的必要条件是 $G\succeq0$。用正交变换 $(u,v)\mapsto(u+v,u-v)$ 可把二次型分解为

$$
(u+v)^\top(2\Sigma-D_s)(u+v)
+(u-v)^\top D_s(u-v),
$$

故需 $D_s\succeq0$ 且 $2\Sigma-D_s\succeq0$。

### B. 反对称统计量

令 $Z_j$、$\widetilde Z_j$ 分别衡量原变量与替身的重要性，定义

$$
W_j=Z_j-\widetilde Z_j.
$$

交换第 $j$ 对变量后，两项互换，因此

$$
W_j^{\mathrm{swap}}=\widetilde Z_j-Z_j=-W_j.
$$

这一步只用减法反对称性，是后续把负统计量当作假发现镜像计数的基础。

### C. 阈值

定义正发现数 $R(t)=\#\{j:W_j\ge t\}$，镜像负数 $L(t)=\#\{j:W_j\le-t\}$。Knockoff+ 阈值为

$$
T=\min\left\{t>0:
\frac{1+L(t)}{\max(1,R(t))}\le q
\right\}.
$$

分子加一提供有限样本保守修正，分母取最大值避免除零。最终集合是 $\widehat S=\{j:W_j\ge T\}$；FDR 控制还依赖交换性等概率论条件，而非仅由阈值代数形式保证。

## 5. 把推导压缩成最终公式

前一节给出了公式的来源。本节把最终计算所需的关系集中列出，便于实现时逐项对应。

### 5.1 1. Gram 约束

$$
\tilde X^\top\tilde X=X^\top X=\Sigma,\qquad X^\top\tilde X=\Sigma-\operatorname{diag}(s).
$$

### 5.2 2. 竞争统计量

为每对原变量和 knockoff 构造反对称重要度。

$$
W_j=Z_j-\tilde Z_j,\qquad W_j([X_j,\tilde X_j]\text{ swapped})=-W_j.
$$

### 5.3 3. 数据驱动阈值

$$
T=\min\left\{t>0:\frac{1+\#\{j:W_j\le-t\}}{\max(1,\#\{j:W_j\ge t\})}\le q\right\}.
$$

### 5.4 4. 选择

$$
\hat S=\{j:W_j\ge T\}.
$$

## 6. 从公式到算法

**输入：** 固定设计矩阵 $X\in\mathbb R^{n\times p}$，构造 knockoff 矩阵 $\tilde X$，要求交换原变量与对应 knockoff 后 Gram 结构不变。
**输出：** 带有限样本 FDR 控制的特征集合。

1. 从 Gram 约束构造 knockoff 特征。
2. 在增广设计上拟合模型并计算成对重要度。
3. 形成反对称统计量。
4. 按目标 FDR 计算阈值并选择。

上面的顺序直接来自前述推导：先计算定义算法状态所需的统计量，再求闭式解、执行递推或更新优化变量，最后按算法定义读取输出。这里没有加入独立于目标函数的额外规则。

## 7. 最小可复算例子

假设统计量 $W=(4,2,-3,0.5)$，目标 $q=0.5$。候选阈值 $t=2$ 时，正侧个数 $R=2$，负侧个数 $L=1$。Knockoff+ 比率

$$
\frac{1+L}{R}=\frac22=1>0.5,
$$

所以不能在 $t=2$ 选择。若普通 knockoff 不加 $1$，比率为 $1/2$，则可选 $W\ge2$ 的两个变量。

## 8. 如何解释结果

本算法输出所表达的是“在控制错误发现率的前提下进行变量选择”这一特定数学目标，而不是对数据全部结构的完整描述。解释结果时应直接回到目标函数、距离、概率或集合定义。

- 固定设计版本要求适当的样本维数和 Gram 可行性。

## 9. 计算复杂度

包含 knockoff 构造和一次增广模型拟合。
