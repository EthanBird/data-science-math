# 影响函数（Influence Functions）：数学推导 Tutorial

> 首次提出或经典年份：2017  
> 类别：解释与影响分析  
> 本章目标：一阶近似单个训练样本对参数和测试预测的影响  
> 先修知识：条件期望、局部回归、合作博弈与微分

## 1. 问题从哪里来

模型解释不是模型的固有唯一属性，它还依赖背景分布、扰动方式、价值函数和损失。每种解释方法都必须先固定“解释对象”，再推导贡献、效应或影响量。

**影响函数（Influence Functions）**要解决的具体问题是：一阶近似单个训练样本对参数和测试预测的影响。本章不从最终公式开始，而是先定义数据对象和希望保留的关系，再逐步得到公式与算法。

### 1.1 本章将得到什么

- 推导训练样本加权扰动
- 说明最终公式与算法步骤之间的对应关系
- 用一个最小例子检查推导结果

## 2. 数据、符号与假设

经验风险 $R(\theta)=n^{-1}\sum_iL(z_i,\theta)$，最优参数 $\hat\theta$ 满足 $\nabla R(\hat\theta)=0$。

算法输出为：训练样本对参数或测试损失的有符号影响分数。

为了使上面的数学对象有定义，需要以下前提：

- 模型 $f$ 固定，背景分布或扰动分布明确。
- 解释结论只针对所选价值函数、邻域和损失。

## 3. 建模前的基础数学

### 3.1 解释量依赖哪些对象

给定模型 $f$、数据分布 $P_X$ 和样本 $x$，解释一般写成

$$
E=\mathcal E(f,P_X,x).
$$

改变背景分布、条件化方式或损失函数，会得到不同解释。因此在推导贡献前，必须先固定价值函数或扰动分布。

### 3.2 局部线性化

若 $f$ 在 $x$ 附近可微，则

$$
f(x+h)=f(x)+\nabla f(x)^\top h+O(\|h\|_2^2).
$$

梯度解释只在高阶余项足够小时可靠；LIME、ALE、DALE 和影响函数都可视为在不同对象上的局部近似。

## 4. 从定义到算法的完整推导

### A. 训练样本加权扰动

经验风险

$$
R(\theta)=\frac1n\sum_i\ell_i(\theta)
$$

在最优点满足 $\nabla R(\widehat\theta)=0$。给样本 $z$ 增加微小权重 $\varepsilon$：

$$
R_\varepsilon(\theta)=R(\theta)+\varepsilon\ell_z(\theta).
$$

新最优点满足

$$
F(\theta,\varepsilon)=\nabla R(\theta)+
\varepsilon\nabla\ell_z(\theta)=0.
$$

对 $\varepsilon$ 隐式求导：

$$
H_{\widehat\theta}\frac{d\widehat\theta_\varepsilon}{d\varepsilon}
+\nabla\ell_z(\widehat\theta)=0.
$$

若 Hessian $H$ 可逆，

$$
\left.\frac{d\widehat\theta_\varepsilon}{d\varepsilon}\right|_0
=-H^{-1}\nabla\ell_z(\widehat\theta).
$$

测试损失的影响再用链式法则左乘测试梯度。

## 5. 把推导压缩成最终公式

前一节给出了公式的来源。本节把最终计算所需的关系集中列出，便于实现时逐项对应。

### 5.1 1. 上调样本权重

对样本 $z$ 增加无穷小权重 $\varepsilon$。

$$
\hat\theta_{\varepsilon,z}=\arg\min_\theta R(\theta)+\varepsilon L(z,\theta).
$$

### 5.2 2. 隐函数求导

对一阶条件在 $\varepsilon=0$ 求导。

$$
H_{\hat\theta}\frac{d\hat\theta_{\varepsilon,z}}{d\varepsilon}+\nabla_\theta L(z,\hat\theta)=0.
$$

### 5.3 3. 参数影响

$$
\mathcal I_{\mathrm{up,param}}(z)=-H_{\hat\theta}^{-1}\nabla_\theta L(z,\hat\theta).
$$

### 5.4 4. 测试损失影响

再乘测试损失梯度。

$$
\mathcal I_{\mathrm{up,loss}}(z,z_{\mathrm{test}})=-\nabla L(z_{\mathrm{test}},\hat\theta)^\top H_{\hat\theta}^{-1}\nabla L(z,\hat\theta).
$$

### 5.5 5. 删除样本近似

删除一条样本约等于权重变化 $-1/n$。

$$
\hat\theta_{-i}-\hat\theta\approx-\frac1n\mathcal I_{\mathrm{up,param}}(z_i).
$$

## 6. 从公式到算法

**输入：** 经验风险 $R(\theta)=n^{-1}\sum_iL(z_i,\theta)$，最优参数 $\hat\theta$ 满足 $\nabla R(\hat\theta)=0$。
**输出：** 训练样本对参数或测试损失的有符号影响分数。

1. 计算训练最优点的梯度与 Hessian 向量积。
2. 用共轭梯度求 $H^{-1}v$ 而非显式求逆。
3. 对每个训练样本计算内积影响。
4. 用实际重训抽样验证一阶近似。

上面的顺序直接来自前述推导：先计算定义算法状态所需的统计量，再求闭式解、执行递推或更新优化变量，最后按算法定义读取输出。这里没有加入独立于目标函数的额外规则。

## 7. 最小可复算例子

经验风险最优点满足 $\nabla R(\hat\theta)=0$。把训练样本 $z$ 权重增加 $\varepsilon$ 后，参数变化一阶近似

$$
\frac{d\hat\theta}{d\varepsilon}=-H_{\hat\theta}^{-1}\nabla_\theta\ell(z,\hat\theta).
$$

若测试损失梯度为 $g_{\text{test}}$，该训练样本对测试损失的影响为 $-g_{\text{test}}^\top H^{-1}\nabla\ell(z)$。正值表示提高样本权重会使测试损失上升。

## 8. 如何解释结果

本算法输出所表达的是“一阶近似单个训练样本对参数和测试预测的影响”这一特定数学目标，而不是对数据全部结构的完整描述。解释结果时应直接回到目标函数、距离、概率或集合定义。

- 要求局部二阶可微且 Hessian 可逆或适当阻尼。
- 大幅删除时一阶近似可能失效。

## 9. 计算复杂度

主要为若干 Hessian-向量乘与线性系统求解。
