# 拉普拉斯特征映射（Laplacian Eigenmaps）：数学推导 Tutorial

> 首次提出或经典年份：2003  
> 类别：图、拓扑与流形结构  
> 本章目标：保持邻接图中相近样本在低维仍相近  
> 先修知识：图论、马尔可夫链、谱分解与流形

## 1. 问题从哪里来

流形与图方法先把样本转换成邻接图，再通过最短路、随机游走、图拉普拉斯或模糊集合构造低维表示。数据几何因此被编码在边和权重中。

**拉普拉斯特征映射（Laplacian Eigenmaps）**要解决的具体问题是：保持邻接图中相近样本在低维仍相近。本章不从最终公式开始，而是先定义数据对象和希望保留的关系，再逐步得到公式与算法。

### 1.1 本章将得到什么

- 推导邻域保持目标
- 说明最终公式与算法步骤之间的对应关系
- 用一个最小例子检查推导结果

## 2. 数据、符号与假设

相似图 $W$、度矩阵 $D$ 与拉普拉斯 $L=D-W$。低维坐标矩阵 $Y\in\mathbb R^{n\times d}$。

算法输出为：保持局部图关系的低维坐标。

为了使上面的数学对象有定义，需要以下前提：

- 邻接图构造、边权和连通分量规则固定。
- 图不连通或度为零的顶点需要单独处理。

## 3. 建模前的基础数学

### 3.1 从样本构造图

令相似度矩阵 $W=(w_{ij})$ 满足 $w_{ij}\ge0$，度矩阵为

$$
D_{ii}=\sum_jw_{ij},
$$

图拉普拉斯为 $L=D-W$。若图无向，则

$$
\begin{aligned}
f^\top Lf
&=\sum_iD_{ii}f_i^2-\sum_{i,j}w_{ij}f_if_j\\
&=\frac12\sum_{i,j}w_{ij}(f_i-f_j)^2.
\end{aligned}
$$

因此最小化 $f^\top Lf$ 会使相邻顶点的表示接近。

## 4. 从定义到算法的完整推导

### A. 邻域保持目标

希望相似点坐标接近，定义

$$
J(Y)=\frac12\sum_{i,j}w_{ij}\|y_i-y_j\|^2.
$$

逐坐标应用图拉普拉斯恒等式：

$$
J(Y)=\operatorname{tr}(Y^\top L Y).
$$

为排除全零解，加入尺度约束 $Y^\top DY=I$；为排除常数特征向量，加入 $Y^\top D\mathbf1=0$。拉格朗日条件给出

$$
LY=DY\Lambda.
$$

因此嵌入坐标由广义特征问题 $Lv=\lambda Dv$ 的最小非零特征向量组成。

### B. 把邻域平方差完整化为迹形式

把 $Y$ 的第 $r$ 列记为 $y^{(r)}=(Y_{1r},\ldots,Y_{nr})^\top$。从欧氏范数定义出发，

$$
\begin{aligned}
J(Y)
&=\frac12\sum_{i,j}w_{ij}\sum_{r=1}^d(Y_{ir}-Y_{jr})^2\\
&=\sum_{r=1}^d\frac12\sum_{i,j}w_{ij}
\left(Y_{ir}^2+Y_{jr}^2-2Y_{ir}Y_{jr}\right).
\end{aligned}
$$

因为图无向，$w_{ij}=w_{ji}$，前两个平方项相等，并且 $D_{ii}=\sum_jw_{ij}$，所以

$$
\begin{aligned}
J(Y)
&=\sum_{r=1}^d\left[
\sum_iD_{ii}Y_{ir}^2-\sum_{i,j}w_{ij}Y_{ir}Y_{jr}
\right]\\
&=\sum_{r=1}^d (y^{(r)})^\top(D-W)y^{(r)}\\
&=\sum_{r=1}^d (y^{(r)})^\top Ly^{(r)}\\
&=\operatorname{tr}(Y^\top LY).
\end{aligned}
$$

第一行只展开平方；第二行代入度矩阵定义；最后一行使用迹等于对角元素之和。

### C. 从矩阵拉格朗日函数推出广义特征方程

约束 $Y^\top DY=I_d$ 固定加权尺度。构造

$$
\mathcal L(Y,\Lambda)
=\operatorname{tr}(Y^\top LY)
-\operatorname{tr}\left[\Lambda(Y^\top DY-I_d)\right],
$$

其中 $\Lambda=\Lambda^\top$。由于 $L,D$ 均对称，矩阵微分为

$$
d\operatorname{tr}(Y^\top LY)=2\operatorname{tr}[(LY)^\top dY],
$$

$$
d\operatorname{tr}(\Lambda Y^\top DY)
=2\operatorname{tr}[(DY\Lambda)^\top dY].
$$

因此

$$
d\mathcal L
=2\operatorname{tr}[(LY-DY\Lambda)^\top dY].
$$

$dY$ 可以任意变化，一阶条件只能是

$$
LY=DY\Lambda.
$$

这说明 $Y$ 的每一列都是 $Lv=\lambda Dv$ 的广义特征向量。又因为

$$
L\mathbf1=(D-W)\mathbf1=D\mathbf1-W\mathbf1=0,
$$

常数向量对应 $\lambda=0$，只会把所有点映射到同一坐标，必须跳过。根据广义 Rayleigh–Ritz 原理，剩余最小的非零特征值给出最小平滑能量。

## 5. 把推导压缩成最终公式

前一节给出了公式的来源。本节把最终计算所需的关系集中列出，便于实现时逐项对应。

### 5.1 1. 局部平滑能量

大边权连接的点应接近。

$$
J(Y)=\frac12\sum_{i,j}W_{ij}\lVert y_i-y_j\rVert_2^2=\operatorname{tr}(Y^\top LY).
$$

### 5.2 2. 去除塌缩解

固定加权尺度与中心。

$$
Y^\top DY=I_d,\qquad Y^\top D\mathbf1=0.
$$

### 5.3 3. 广义特征方程

拉格朗日条件。

$$
Ly=\lambda Dy.
$$

### 5.4 4. 坐标选择

跳过常数特征向量，取最小非零特征值方向。

$$
Y=[y_2,\ldots,y_{d+1}].
$$

## 6. 从公式到算法

**输入：** 相似图 $W$、度矩阵 $D$ 与拉普拉斯 $L=D-W$。低维坐标矩阵 $Y\in\mathbb R^{n\times d}$。
**输出：** 保持局部图关系的低维坐标。

1. 构造局部邻接图和权重。
2. 形成 $L,D$。
3. 求最小广义特征对。
4. 去掉零特征值常数方向。

上面的顺序直接来自前述推导：先计算定义算法状态所需的统计量，再求闭式解、执行递推或更新优化变量，最后按算法定义读取输出。这里没有加入独立于目标函数的额外规则。

## 7. 最小可复算例子

三节点路径图权重均为 $1$，

$$
L=\begin{bmatrix}1&-1&0\\-1&2&-1\\0&-1&1\end{bmatrix},
\quad D=\operatorname{diag}(1,2,1).
$$

常数向量满足 $L\mathbf1=0$。解广义特征问题 $Lv=\lambda Dv$，最小非零方向可取 $v=(-1,0,1)^\top$，它把路径两端放在两侧、中点放在中央，保持局部邻接顺序。

## 8. 如何解释结果

本算法输出所表达的是“保持邻接图中相近样本在低维仍相近”这一特定数学目标，而不是对数据全部结构的完整描述。解释结果时应直接回到目标函数、距离、概率或集合定义。

- 断开的图会产生多个零特征值。

## 9. 计算复杂度

稀疏图上使用迭代特征求解。
