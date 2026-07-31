# Data Science Math：数学推导 Tutorial

本库把数据科学与机器学习中常用的数据处理算法写成可连续阅读的数学教程。每篇遵循同一学习顺序：问题背景 → 数据与符号 → 所需数学 → 逐步推导 → 算法步骤 → 专属手算例子 → 结果解释。

文件按经典提出年份命名；没有明确单一年份的基础方法使用 `0000-` 前缀。全部公式使用 GitHub/KaTeX 兼容语法。

- 算法 Tutorial：**150 篇**
- 每个算法独立一个 Markdown 文件
- 不依赖大模型内容，重点是确定性、可解释的数据处理数学

## 阅读路径

建议先阅读基础变换、距离与矩阵分解，再进入降维、聚类、时间序列和解释性方法。每篇均可独立阅读。

## 目录

### 基础数据构造

- [基础方法（无唯一首发年份） · 截断（Clipping）](00-foundations/0000-clipping.md) — 把数值限制在闭区间内
- [基础方法（无唯一首发年份） · 周期编码（Cyclical Encoding）](00-foundations/0000-cyclical-encoding.md) — 把周期变量嵌入单位圆，使周期端点相邻
- [基础方法（无唯一首发年份） · 效应与对比编码（Effect / Contrast Coding）](00-foundations/0000-effect-contrast-coding.md) — 用线性独立对比表示类别相对总体或基准的效应
- [基础方法（无唯一首发年份） · 交互特征（Interaction Features）](00-foundations/0000-interaction-features.md) — 显式表示变量联合变化而非单独加和的效应
- [基础方法（无唯一首发年份） · 独热编码（One-hot Encoding）](00-foundations/0000-one-hot-encoding.md) — 把无序类别映射到标准基向量
- [基础方法（无唯一首发年份） · 有序编码（Ordinal Encoding）](00-foundations/0000-ordinal-encoding.md) — 把有序类别嵌入到整数序列
- [基础方法（无唯一首发年份） · 多项式特征（Polynomial Features）](00-foundations/0000-polynomial-features.md) — 把低维输入显式嵌入有限阶多项式空间
- [基础方法（无唯一首发年份） · 单位与量纲统一（Unit and Dimension Normalization）](00-foundations/0000-unit-dimension-normalization.md) — 在保持物理量不变的前提下统一数值表示
- [2009 · 特征哈希（Feature Hashing）](00-foundations/2009-feature-hashing.md) — 用固定哈希把高维稀疏特征压缩到固定维数

### 数据质量与稳健清洗

- [基础方法（无唯一首发年份） · 规则约束校验（Constraint Validation）](01-data-quality/0000-constraint-validation.md) — 把数据质量要求写成可判定的数学谓词
- [基础方法（无唯一首发年份） · 精确去重与哈希去重（Exact / Hash Deduplication）](01-data-quality/0000-exact-hash-deduplication.md) — 把完全相同或键等价的记录压缩为一个代表元
- [基础方法（无唯一首发年份） · 温莎化（Winsorization）](01-data-quality/0000-winsorization.md) — 用经验分位点替代极端尾部值
- [1981 · 随机采样一致性（RANSAC）](01-data-quality/1981-ransac.md) — 在含大量离群点的数据中寻找最大一致内点集合

### 缩放、变换与离散化

- [基础方法（无唯一首发年份） · 等频分箱（Equal-frequency Binning）](02-scaling-transforms/0000-equal-frequency-binning.md) — 用经验分位点构造近似等样本量区间
- [基础方法（无唯一首发年份） · 等宽分箱（Equal-width Binning）](02-scaling-transforms/0000-equal-width-binning.md) — 按固定数值宽度把连续变量离散化
- [基础方法（无唯一首发年份） · 对数与 log1p 变换（Log / log1p Transform）](02-scaling-transforms/0000-log1p-transform.md) — 压缩长尾并把乘法结构转为加法结构
- [基础方法（无唯一首发年份） · 最大绝对值缩放（MaxAbs Scaling）](02-scaling-transforms/0000-maxabs-scaling.md) — 在不平移数据的情况下把绝对值缩放到不超过一
- [基础方法（无唯一首发年份） · Min-Max 归一化（Min-Max Scaling）](02-scaling-transforms/0000-min-max-scaling.md) — 用仿射变换把每列映射到给定闭区间
- [基础方法（无唯一首发年份） · 中位数-IQR 稳健缩放（Median-IQR Robust Scaling）](02-scaling-transforms/0000-robust-scaling.md) — 以分位数统计量完成对异常值不敏感的中心化与缩放
- [基础方法（无唯一首发年份） · 单位范数归一化（Unit Norm Normalization）](02-scaling-transforms/0000-unit-norm-normalization.md) — 把每个样本投影到单位球面
- [基础方法（无唯一首发年份） · Z-score 标准化（Z-score Standardization）](02-scaling-transforms/0000-z-score-standardization.md) — 把每列平移到零均值并缩放到单位标准差
- [1955 · 保序回归（Isotonic Regression）](02-scaling-transforms/1955-isotonic-regression.md) — 在单调约束下得到最小平方误差拟合
- [1964 · Box-Cox 变换（Box-Cox Transform）](02-scaling-transforms/1964-box-cox-transform.md) — 通过参数化幂变换稳定方差并改善正态近似
- [1992 · ChiMerge 监督分箱（ChiMerge）](02-scaling-transforms/1992-chimerge.md) — 用相邻区间的类别分布卡方差异决定合并
- [1993 · 最小描述长度离散化（MDLP Discretization）](02-scaling-transforms/1993-mdlp-discretization.md) — 以信息增益和描述长度准则递归确定监督分割点
- [2000 · Yeo-Johnson 变换（Yeo-Johnson Transform）](02-scaling-transforms/2000-yeo-johnson-transform.md) — 把 Box-Cox 幂变换扩展到零值和负值
- [2003 · 分位数归一化（Quantile Normalization）](02-scaling-transforms/2003-quantile-normalization.md) — 使多个样本列具有完全相同的经验边际分布
- [2004 · CAIM 离散化（Class-Attribute Interdependence Maximization）](02-scaling-transforms/2004-caim-discretization.md) — 寻找使分箱与类别依赖性最大的监督边界

### 缺失数据

- [基础方法（无唯一首发年份） · 均值填补（Mean Imputation）](03-missing-data/0000-mean-imputation.md) — 用观测均值作为平方损失下的常数填补值
- [基础方法（无唯一首发年份） · 中位数填补（Median Imputation）](03-missing-data/0000-median-imputation.md) — 用绝对损失下的最优常数稳健填补缺失值
- [基础方法（无唯一首发年份） · 缺失指示变量（Missingness Indicator）](03-missing-data/0000-missing-indicator.md) — 把缺失机制本身编码为显式二元特征
- [基础方法（无唯一首发年份） · 众数填补（Mode Imputation）](03-missing-data/0000-mode-imputation.md) — 以零一损失下最优的类别常数填补缺失类别
- [1977 · 期望最大化算法（Expectation-Maximization Algorithm）](03-missing-data/1977-expectation-maximization.md) — 在隐变量或不完整数据下迭代最大化观测似然
- [2001 · KNN 填补（K-nearest-neighbour Imputation）](03-missing-data/2001-knn-imputation.md) — 用在共同观测维度上最相似的样本估计缺失值
- [2010 · SoftImpute（SoftImpute）](03-missing-data/2010-softimpute.md) — 用奇异值软阈值迭代求解低秩缺失矩阵填补
- [2012 · MissForest（MissForest）](03-missing-data/2012-missforest.md) — 用逐列随机森林迭代填补数值与类别混合缺失值

### 距离、核与对齐

- [1936 · 马氏距离（Mahalanobis Distance）](04-distance-alignment/1936-mahalanobis-distance.md) — 按协方差几何测量多变量距离
- [1971 · Gower 距离（Gower Distance）](04-distance-alignment/1971-gower-distance.md) — 统一度量数值、类别、有序和二元混合变量
- [1978 · 动态时间规整（Dynamic Time Warping）](04-distance-alignment/1978-dynamic-time-warping.md) — 在保持时间顺序的条件下非线性对齐两条序列
- [2017 · Soft-DTW（Soft Dynamic Time Warping）](04-distance-alignment/2017-soft-dtw.md) — 用平滑最小算子把 DTW 变为可微损失
- [2024 · 自动 Gower 加权（Automatic Gower Weighting）](04-distance-alignment/2024-automatic-gower-weighting.md) — 从监督或结构目标学习混合数据 Gower 距离中的变量权重

### 特征选择与稀疏化

- [1992 · Relief 特征评分（Relief）](05-feature-selection/1992-relief.md) — 根据最近同类与异类样本的差异更新特征重要性
- [1994 · ReliefF 特征评分（ReliefF）](05-feature-selection/1994-relieff.md) — 把 Relief 扩展到多类别、多个邻居和含噪数据
- [1996 · LASSO（Least Absolute Shrinkage and Selection Operator）](05-feature-selection/1996-lasso.md) — 用 $L_1$ 惩罚同时估计和稀疏选择线性特征
- [1999 · 相关性特征选择（Correlation-based Feature Selection）](05-feature-selection/1999-correlation-feature-selection.md) — 选择与目标相关但彼此冗余较低的特征子集
- [2005 · Elastic Net（Elastic Net）](05-feature-selection/2005-elastic-net.md) — 结合 $L_1$ 稀疏性和 $L_2$ 稳定性选择相关特征组
- [2005 · 融合 LASSO（Fused Lasso）](05-feature-selection/2005-fused-lasso.md) — 同时得到稀疏且相邻系数分段平滑的解
- [2005 · 最小冗余最大相关（Minimum Redundancy Maximum Relevance）](05-feature-selection/2005-mrmr.md) — 选择与目标信息量大、彼此重复信息少的特征
- [2006 · 组 LASSO（Group Lasso）](05-feature-selection/2006-group-lasso.md) — 按预定义特征组整体选择或删除变量
- [2010 · Boruta（Boruta Feature Selection）](05-feature-selection/2010-boruta.md) — 通过影子特征检验找出所有相关而非最小相关特征集
- [2010 · 稳定性选择（Stability Selection）](05-feature-selection/2010-stability-selection.md) — 通过重复子采样量化特征被选择的稳定概率
- [2015 · Knockoff 过滤器（Knockoff Filter）](05-feature-selection/2015-knockoff-filter.md) — 在控制错误发现率的前提下进行变量选择
- [2018 · Model-X Knockoffs（Model-X Knockoffs）](05-feature-selection/2018-model-x-knockoffs.md) — 在已知或可估计特征分布下控制变量选择 FDR
- [2024 · DELVE（Dynamic selection of locally covarying features）](05-feature-selection/2024-delve.md) — 选择能保留局部图结构与动态轨迹信息的特征

### 降维与表示

- [1901 · 主成分分析（Principal Component Analysis）](06-dimensionality-reduction/1901-principal-component-analysis.md) — 寻找保留最大方差的正交线性子空间
- [1936 · 典型相关分析（Canonical Correlation Analysis）](06-dimensionality-reduction/1936-canonical-correlation-analysis.md) — 寻找两组变量间相关性最大的成对线性投影
- [1936 · 线性判别分析（Linear Discriminant Analysis）](06-dimensionality-reduction/1936-linear-discriminant-analysis.md) — 寻找类间离散大、类内离散小的监督投影
- [1966 · 偏最小二乘（Partial Least Squares）](06-dimensionality-reduction/1966-partial-least-squares.md) — 提取与响应变量协方差最大的潜变量
- [1984 · Johnson-Lindenstrauss 随机投影（Johnson-Lindenstrauss Random Projection）](06-dimensionality-reduction/1984-johnson-lindenstrauss-random-projection.md) — 用较低维随机线性映射近似保持有限样本间距离
- [1997 · FastICA（Fast Independent Component Analysis）](06-dimensionality-reduction/1997-fastica.md) — 通过最大化非高斯性分离统计独立源信号
- [1998 · 核主成分分析（Kernel PCA）](06-dimensionality-reduction/1998-kernel-pca.md) — 在隐式特征空间中执行非线性主成分分析
- [2003 · 稀疏主成分分析（Sparse PCA）](06-dimensionality-reduction/2003-sparse-pca.md) — 让主成分载荷只依赖少量原始变量
- [2008 · t-SNE（t-distributed Stochastic Neighbor Embedding）](06-dimensionality-reduction/2008-t-sne.md) — 通过匹配高低维邻域概率突出局部结构
- [2019 · TriMap（TriMap）](06-dimensionality-reduction/2019-trimap.md) — 通过三元组相对距离约束兼顾局部和全局结构
- [2021 · PaCMAP（Pairwise Controlled Manifold Approximation）](06-dimensionality-reduction/2021-pacmap.md) — 用近邻、中近邻和远点三类配对平衡局部与全局结构

### 聚类

- [1963 · Ward 层次聚类（Ward Hierarchical Clustering）](07-clustering/1963-ward-hierarchical-clustering.md) — 每次合并使类内平方和增加最少的两个簇
- [1987 · PAM 与 K-medoids（PAM / K-medoids）](07-clustering/1987-pam-k-medoids.md) — 以真实样本作为中心最小化簇内距离和
- [1996 · BIRCH（Balanced Iterative Reducing and Clustering using Hierarchies）](07-clustering/1996-birch.md) — 用可增量聚类特征树压缩大规模数据
- [1996 · DBSCAN（Density-Based Spatial Clustering of Applications with Noise）](07-clustering/1996-dbscan.md) — 用局部密度连通性发现任意形状簇并标记噪声
- [1997 · K-prototypes（K-prototypes）](07-clustering/1997-k-prototypes.md) — 联合聚类数值变量与类别变量
- [1999 · OPTICS（Ordering Points To Identify the Clustering Structure）](07-clustering/1999-optics.md) — 用可达距离排序揭示多密度层次聚类结构
- [2002 · 谱聚类（Spectral Clustering）](07-clustering/2002-spectral-clustering.md) — 把图切割问题转化为图拉普拉斯特征向量问题
- [2007 · 亲和传播（Affinity Propagation）](07-clustering/2007-affinity-propagation.md) — 通过样本间消息传递选择真实样本作为代表点
- [2013 · HDBSCAN*（Hierarchical DBSCAN）](07-clustering/2013-hdbscan.md) — 通过互达距离层次树发现不同密度的稳定簇

### 异常检测

- [1974 · Hampel 识别器与滤波（Hampel Identifier / Filter）](08-anomaly-detection/1974-hampel-filter.md) — 用局部中位数和 MAD 检测时间窗口内的异常值
- [1984 · 最小协方差行列式（Minimum Covariance Determinant）](08-anomaly-detection/1984-minimum-covariance-determinant.md) — 通过最紧致子样本稳健估计位置与协方差
- [2000 · 局部离群因子（Local Outlier Factor）](08-anomaly-detection/2000-local-outlier-factor.md) — 比较样本与其邻域的局部密度以发现局部异常
- [2001 · 单类支持向量机（One-Class SVM）](08-anomaly-detection/2001-one-class-svm.md) — 在特征空间中学习包含多数正常样本的紧致区域
- [2008 · 孤立森林（Isolation Forest）](08-anomaly-detection/2008-isolation-forest.md) — 用随机切分所需路径长度衡量样本被孤立的容易程度
- [2020 · COPOD（Copula-Based Outlier Detection）](08-anomaly-detection/2020-copod.md) — 用经验 Copula 的多维尾概率构造无参数异常分数
- [2022 · ECOD（Empirical Cumulative Distribution-based Outlier Detection）](08-anomaly-detection/2022-ecod.md) — 用各维经验分布尾概率构造无参数异常分数

### 不平衡学习与采样

- [1972 · 编辑最近邻（Edited Nearest Neighbours）](09-imbalanced-sampling/1972-edited-nearest-neighbours.md) — 删除局部邻域中标签不一致的疑似噪声或边界样本
- [1976 · Tomek 链（Tomek Links）](09-imbalanced-sampling/1976-tomek-links.md) — 识别类别边界上互为最近邻的异类样本对
- [2002 · SMOTE（Synthetic Minority Over-sampling Technique）](09-imbalanced-sampling/2002-smote.md) — 在少数类近邻线段上插值生成合成样本
- [2005 · Borderline-SMOTE（Borderline-SMOTE）](09-imbalanced-sampling/2005-borderline-smote.md) — 优先在分类边界附近的少数类危险样本周围插值
- [2008 · ADASYN（Adaptive Synthetic Sampling）](09-imbalanced-sampling/2008-adasyn.md) — 按局部分类难度自适应分配少数类合成样本数

### 时间序列与信号处理

- [基础方法（无唯一首发年份） · 差分（Differencing）](10-time-series-signal/0000-differencing.md) — 用离散导数消除低阶趋势并表达变化量
- [基础方法（无唯一首发年份） · 滞后特征（Lag Features）](10-time-series-signal/0000-lag-features.md) — 把历史观测转换为当前时刻可用的有序特征
- [基础方法（无唯一首发年份） · 滚动统计（Rolling Statistics）](10-time-series-signal/0000-rolling-statistics.md) — 用固定历史窗口构造局部统计特征
- [1960 · 卡尔曼滤波（Kalman Filter）](10-time-series-signal/1960-kalman-filter.md) — 在线估计线性高斯动态系统的隐藏状态
- [1964 · Savitzky-Golay 滤波（Savitzky-Golay Filter）](10-time-series-signal/1964-savitzky-golay-filter.md) — 用滑动局部多项式最小二乘平滑并估计导数
- [1979 · 局部加权回归（LOWESS / LOESS）](10-time-series-signal/1979-lowess-loess.md) — 以局部低阶多项式形成可解释平滑曲线
- [1988 · 离散小波变换（Discrete Wavelet Transform）](10-time-series-signal/1988-discrete-wavelet-transform.md) — 用多尺度正交或双正交基分解信号
- [1990 · STL 分解（Seasonal-Trend decomposition using LOESS）](10-time-series-signal/1990-stl-decomposition.md) — 把序列分为趋势、季节项与残差
- [1992 · 全变差去噪（Total Variation Denoising）](10-time-series-signal/1992-total-variation-denoising.md) — 在保留阶跃边缘的同时抑制高频噪声
- [1995 · 小波阈值去噪（Wavelet Thresholding）](10-time-series-signal/1995-wavelet-thresholding.md) — 在小波域收缩噪声主导的小系数
- [1998 · 经验模态分解（Empirical Mode Decomposition）](10-time-series-signal/1998-empirical-mode-decomposition.md) — 自适应把非平稳信号分解为若干本征模态函数
- [2007 · 贝叶斯在线变点检测（Bayesian Online Change Point Detection）](10-time-series-signal/2007-bayesian-online-change-point-detection.md) — 在线维护当前片段长度后验并检测结构变化
- [2012 · PELT 变点检测（Pruned Exact Linear Time）](10-time-series-signal/2012-pelt.md) — 精确最小化带惩罚的多变点分段代价并剪枝候选
- [2012 · 小波散射变换（Wavelet Scattering Transform）](10-time-series-signal/2012-wavelet-scattering-transform.md) — 用固定小波、模长和平均构造平移稳定的多尺度特征
- [2014 · 趋势滤波（Trend Filtering）](10-time-series-signal/2014-trend-filtering.md) — 用高阶差分稀疏性估计分段多项式趋势
- [2014 · 变分模态分解（Variational Mode Decomposition）](10-time-series-signal/2014-variational-mode-decomposition.md) — 把信号分解为若干具有有限带宽和中心频率的模态
- [2014 · 野生二分分割（Wild Binary Segmentation）](10-time-series-signal/2014-wild-binary-segmentation.md) — 通过大量随机子区间提高密集多变点检测能力
- [2016 · 矩阵轮廓（Matrix Profile）](10-time-series-signal/2016-matrix-profile.md) — 记录每个时间子序列与其最近非平凡邻居的距离
- [2016 · tsfresh 特征提取与筛选（tsfresh）](10-time-series-signal/2016-tsfresh.md) — 批量构造具名时间序列统计量并以假设检验筛选
- [2017 · FLUSS（Fast Low-cost Unipotent Semantic Segmentation）](10-time-series-signal/2017-fluss.md) — 利用矩阵轮廓最近邻弧线密度分割时间序列语义片段
- [2019 · catch22（Canonical Time-series Characteristics）](10-time-series-signal/2019-catch22.md) — 用 22 个互补且具名的统计量概括时间序列动力学
- [2019 · RobustSTL（Robust Seasonal-Trend Decomposition）](10-time-series-signal/2019-robust-stl.md) — 在异常值和复杂趋势下稳健分离趋势、季节与残差
- [2020 · ROCKET（Random Convolutional Kernel Transform）](10-time-series-signal/2020-rocket.md) — 用大量固定随机卷积核把时间序列映射为快速线性可分特征
- [2021 · MiniRocket（MiniRocket）](10-time-series-signal/2021-minirocket.md) — 用高度约束的固定卷积核和 PPV 特征快速表示时间序列
- [2021 · MSTL（Multiple Seasonal-Trend decomposition using LOESS）](10-time-series-signal/2021-mstl.md) — 把具有多个季节周期的序列分解为趋势、多个季节项和残差
- [2021 · 稳健多尺度季节趋势分解（Robust Multi-scale Seasonal-Trend Decomposition）](10-time-series-signal/2021-robust-multiscale-seasonal-trend.md) — 在多个时间尺度上稳健分解季节变化、趋势和异常
- [2022 · MultiRocket（MultiRocket）](10-time-series-signal/2022-multirocket.md) — 在 MiniRocket 基础上联合原序列、差分序列和多种池化统计

### 传统图像特征

- [1994 · 局部二值模式（Local Binary Pattern）](11-image-features/1994-local-binary-pattern.md) — 用邻域灰度相对中心的符号模式描述局部纹理
- [1998 · 双边滤波（Bilateral Filter）](11-image-features/1998-bilateral-filter.md) — 同时按空间接近和灰度相似加权平滑图像
- [2004 · 尺度不变特征变换（Scale-Invariant Feature Transform）](11-image-features/2004-sift.md) — 提取对尺度、旋转和局部光照变化较稳定的图像关键点描述子
- [2005 · 方向梯度直方图（Histogram of Oriented Gradients）](11-image-features/2005-histogram-oriented-gradients.md) — 用局部梯度方向分布描述物体轮廓与形状

### 神经网络归一化

- [2015 · 批归一化（Batch Normalization）](12-neural-normalization/2015-batch-normalization.md) — 用小批次统计量标准化中间激活并学习恢复尺度
- [2016 · 层归一化（Layer Normalization）](12-neural-normalization/2016-layer-normalization.md) — 在单个样本内部跨特征归一化激活
- [2018 · 组归一化（Group Normalization）](12-neural-normalization/2018-group-normalization.md) — 在单个样本内按通道组归一化，不依赖批大小
- [2019 · RMSNorm（Root Mean Square Layer Normalization）](12-neural-normalization/2019-rmsnorm.md) — 仅按均方根缩放激活，保留均值信息并简化层归一化

### 解释与影响分析

- [2015 · 个体条件期望曲线（Individual Conditional Expectation）](13-explainability/2015-individual-conditional-expectation.md) — 展示单个样本预测随一个特征替换变化的轨迹
- [2016 · 累积局部效应（Accumulated Local Effects）](13-explainability/2016-accumulated-local-effects.md) — 在相关特征条件分布下估计平均局部预测效应
- [2016 · LIME（Local Interpretable Model-agnostic Explanations）](13-explainability/2016-lime.md) — 在待解释样本附近拟合稀疏可解释替代模型
- [2017 · 影响函数（Influence Functions）](13-explainability/2017-influence-functions.md) — 一阶近似单个训练样本对参数和测试预测的影响
- [2017 · SHAP（SHapley Additive exPlanations）](13-explainability/2017-shap.md) — 按合作博弈公理把预测相对基线分配给各特征
- [2018 · Anchors（Anchors Explanations）](13-explainability/2018-anchors.md) — 寻找在局部扰动分布下具有高精度的 If-Then 规则
- [2018 · TCAV（Testing with Concept Activation Vectors）](13-explainability/2018-tcav.md) — 用用户定义概念方向度量网络输出对概念的敏感性
- [2020 · SAGE（Shapley Additive Global Importance）](13-explainability/2020-sage.md) — 用 Shapley 值分配特征对全局预测性能的贡献
- [2020 · TreeSHAP（TreeSHAP）](13-explainability/2020-treeshap.md) — 利用树结构动态规划高效计算树模型 Shapley 贡献
- [2023 · DALE（Differential Accumulated Local Effects）](13-explainability/2023-dale.md) — 用模型梯度高效估计累积局部效应

### 图、拓扑与流形结构

- [2000 · Isomap（Isometric Mapping）](14-graph-topology/2000-isomap.md) — 近似保持流形上的测地距离进行非线性降维
- [2000 · 局部线性嵌入（Locally Linear Embedding）](14-graph-topology/2000-locally-linear-embedding.md) — 保持每个样本由邻居线性重构的权重关系进行降维
- [2003 · 拉普拉斯特征映射（Laplacian Eigenmaps）](14-graph-topology/2003-laplacian-eigenmaps.md) — 保持邻接图中相近样本在低维仍相近
- [2006 · 扩散映射（Diffusion Maps）](14-graph-topology/2006-diffusion-maps.md) — 用随机游走的多步连通性定义流形坐标
- [2007 · Mapper（Mapper Algorithm）](14-graph-topology/2007-mapper.md) — 把高维数据压缩为反映覆盖、聚类与重叠关系的拓扑图
- [2008 · Louvain 社区发现（Louvain Method）](14-graph-topology/2008-louvain.md) — 通过局部移动和图聚合近似最大化模块度
- [2018 · UMAP（Uniform Manifold Approximation and Projection）](14-graph-topology/2018-umap.md) — 用模糊近邻图和低维交叉熵保持局部流形结构
- [2019 · Leiden 社区发现（Leiden Algorithm）](14-graph-topology/2019-leiden.md) — 在改进模块度/CPM 目标的同时保证社区内部连接质量
- [2019 · PHATE（Potential of Heat-diffusion for Affinity-based Transition Embedding）](14-graph-topology/2019-phate.md) — 以扩散势距离同时表达局部连续性和全局演化结构
- [2021 · densMAP（Density-preserving UMAP）](14-graph-topology/2021-densmap.md) — 在 UMAP 邻域保持目标中加入局部密度保持项
- [2022 · Multiscale PHATE（Multiscale PHATE）](14-graph-topology/2022-multiscale-phate.md) — 在多个分辨率上构造数据状态层级并保持连续几何

### 组成数据

- [1982 · 中心对数比变换（Centered Log-Ratio Transform）](15-compositional-data/1982-centered-log-ratio.md) — 把正组成数据表示为相对几何均值的对数比
- [2003 · 等距对数比变换（Isometric Log-Ratio Transform）](15-compositional-data/2003-isometric-log-ratio.md) — 把组成数据映射到无奇异约束的正交欧氏坐标

### 矩阵分解、最优传输与低秩恢复

- [1967 · Sinkhorn-Knopp 矩阵缩放（Sinkhorn-Knopp Scaling）](16-matrix-factorization/1967-sinkhorn-knopp-scaling.md) — 用对角缩放使正矩阵满足给定行列边际
- [1999 · 非负矩阵分解（Non-negative Matrix Factorization）](16-matrix-factorization/1999-nonnegative-matrix-factorization.md) — 用非负基与非负系数得到部分组成式表示
- [2009 · 核范数矩阵补全（Nuclear-Norm Matrix Completion）](16-matrix-factorization/2009-nuclear-norm-matrix-completion.md) — 用凸低秩代理从部分观测恢复矩阵
- [2011 · Gromov-Wasserstein 对齐（Gromov-Wasserstein Alignment）](16-matrix-factorization/2011-gromov-wasserstein.md) — 仅根据两个空间内部距离结构对齐不同模态样本
- [2011 · 稳健主成分分析（Robust PCA / Principal Component Pursuit）](16-matrix-factorization/2011-robust-pca.md) — 把观测矩阵分解为低秩主体与稀疏异常
- [2013 · 熵正则最优传输（Entropic Optimal Transport）](16-matrix-factorization/2013-entropic-optimal-transport.md) — 用熵正则化快速求两个离散分布间的软匹配
