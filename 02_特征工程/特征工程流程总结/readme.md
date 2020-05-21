# **特征工程目的**
**构造更多质量好的特征**
# **特征工程思考流程**
## 1. 搜索
> 查看数据列名：
> ```python
> print(data_train.columns)
> ```
> 查看数据每列信息（数目，空置和类型）：
> ```python
> print(data_train.info())
> ```
> 查看每列统计信息（数目、均值、方差、最小值、25%分位值、50%分位值、75%分位值和最大值）： 
> ```python
> print(data_train.describe())
> ```

* 特征状况： 
  * 类型：
    * 连续型（均值作为代表）、离散型（众数）
    * int float object 等
    * 考虑是否需要进行离散化
  * 空置：NaN 
    > 利用Pandas库判断数据中是否有空值：
    > ```python
    > print(pd.isnull(df))
    > ```
    > 返回值为等尺寸的DataFram文件，数据类型为：bool类型（True/Flase）   
  * 分布（特征选择）
    > 通过散点图查看单一特征分布： 
    > ```python
    > plt.scatter(X,y)                      #X为特征，y为标签
    > ```
  * 异常点
    > 方差较小的情况下，Max/Min 离均值很远，则Max/Min可能为异常点 <br/>
    > 通过绘图观察，极具偏离大部分样本点的点称为异常点（需要判读该点是数据性质造成的，还是真是存在的异常点）<br/>
    > 异常点的存在，对模型的鲁棒性存在影响
  * 量纲
    > 任意两列数据的单位悬殊过大
* 标签状况：
  * 类型
    > 判断当前问题类型
  * 分布
    > 判断当前是否需要做标签均衡
## 2. 清洗/向量化
* 类别特征编码转为向量
  > 需要对字符串类别的数据做数值编码：
  > * 直接按顺序按转为 0 1 2 （例：a6_titanic\data_process\1_feature.py）
  > * Onehot 编码，解决特征值的距离问题（[1 0 0], [0 1 0], [0 0 1]）（例：a6_titanic\data_process\7_feature_one_hot.py）
* 空值处理
  * 缺失较多，可以丢弃
    ```pyton
    df.dropna()
    ```
  * 填充
    * 填充固定数值
        ```python
        df.fillna(value=0) # NaN → 0
        ```
    * 离散数值 – 众数或新数值
        ```python
        df = data_train.apply(lambda x:x.fillna(x.value_counts().index[0]))
        ```
    * 连续数值 – 均值
        ```python
        df = data_train.apply(lambda x:x.mean(),axis = 1)
        ```
  * ML预测
    > 将已知数据设为数据集，将NaN值设为测试集，进行ML从而得到完整的数据<br/>
    > 示例如下（使用了随机森林进行训练）
    > ```python
    >  def set_missing_ages(df):
    >    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    >    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    >
    >    # 乘客分成已知年龄和未知年龄两部分
    >    known_age = age_df[age_df.Age.notnull()].as_matrix()
    >    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    >
    >    # y即目标年龄
    >    y = known_age[:, 0]
    >
    >    # X即特征属性值
    >    X = known_age[:, 1:]
    >
    >    # fit到RandomForestRegressor之中
    >    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    >    rfr.fit(X, y)
    >
    >    # 用得到的模型进行未知年龄结果预测
    >    predictedAges = rfr.predict(unknown_age[:, 1::])
    >
    >    # 用得到的预测结果填补原缺失数据
    >    df.loc[(df.Age.isnull()), 'Age'] = predictedAges
    >
    >    return df, rfr
    >```
## 3. 标准化
* 异常点
    > 例：a1_numpy_pandas\pandas\3_set_value.py
    > ```python
    > df.A[df.A>70] = 0
    > df.A[df.A<=70] = 1
    > ```
    > 设置取值范围
* 量纲是否需要统一
   - 标准化（也叫Z-score standardization）（对列向量处理）:
        将服从正态分布的特征值转换成标准正态分布，标准化需要计算特征的均值和标准差，公式表达为：
        <br>![](http://images2015.cnblogs.com/blog/927391/201605/927391-20160502113957732-1062097580.png)
        <br>使用preproccessing库的StandardScaler类对数据进行标准化的代码如下：
        ```python
        from sklearn.preprocessing import StandardScaler 
        #标准化，返回值为标准化后的数据
        StandardScaler().fit_transform(iris.data)
        ```
   - 区间缩放（对列向量处理）
    区间缩放法的思路有多种，常见的一种为利用两个最值进行缩放，公式表达为：
    <br>![](http://images2015.cnblogs.com/blog/927391/201605/927391-20160502113301013-1555489078.png)
    <br>使用preproccessing库的MinMaxScaler类对数据进行区间缩放的代码如下：
        ```python
        from sklearn.preprocessing import MinMaxScaler 
        #区间缩放，返回值为缩放到[0, 1]区间的数据
        MinMaxScaler().fit_transform(iris.data)
        ```

* 是否需要归一化
    - 归一化（对行向量处理）
        归一化目的在于样本向量在点乘运算或其他核函数计算相似性时，拥有统一的标准，也就是说都转化为“单位向量”。规则为l2的归一化公式如下：
        <br>![](http://images2015.cnblogs.com/blog/927391/201607/927391-20160719002904919-1602367496.png)
        <br>使用preproccessing库的Normalizer类对数据进行归一化的代码如下：
        ```python
        from sklearn.preprocessing import Normalizer
        #归一化，返回值为归一化后的数据
        Normalizer().fit_transform(iris.data)
        ```
* 分桶离散
    - 对定量特征二值化（对列向量处理）
        **定性与定量区别**
        <br>定性：博主很胖，博主很瘦
        <br>定量：博主有80kg，博主有60kg
        <br>一般定性都会有相关的描述词，定量的描述都是可以用数字来量化处理
        <br><br>定量特征二值化的核心在于设定一个阈值，大于阈值的赋值为1，小于等于阈值的赋值为0，公式表达如下：
        <br>![](http://images2015.cnblogs.com/blog/927391/201605/927391-20160502115121216-456946808.png)
        <br>使用preproccessing库的Binarizer类对数据进行二值化的代码如下：
        ```python
        from sklearn.preprocessing import Binarizer
        #二值化，阈值设置为3，返回值为二值化后的数据
        Binarizer(threshold=3).fit_transform(iris.data)
        ```
        目的：希望数据在特定的区间才存在意义<br>
        例：a2_feature_engineering\6_continue_to_bin\continue_to_discret_bucketing.py
    - Q&A:<br>
        Q: CTR预估，发现CTR预估一般都是用LR，而且特征都是离散的。为什么一定要用离散特征呢？这样做的好处在哪里？<br>
        A: 在工业界，很少直接将连续值作为逻辑回归模型的特征输入，而是将连续特征离散化为一系列0、1特征交给逻辑回归模型，这样做的优势有以下几点：<br>
        - 离散特征的增加和减少都很容易，易于模型的快速迭代。(离散特征的增加和减少，模型也不需要调整，重新训练是必须的，相比贝叶斯推断方法或者树模型方法迭代快)<br>
        - 稀疏向量内积乘法运算速度快，计算结果方便存储，容易扩展；
        - 离散化后的特征对异常数据有很强的鲁棒性：比如一个特征是年龄>30是1，否则0。如果特征没有离散化，一个异常数据“年龄300岁”会给模型造成很大的干扰；离散化后年龄300岁也只对应于一个权重，如果训练数据中没有出现特征"年龄-300岁"，那么在LR模型中，其权重对应于0，所以，即使测试数据中出现特征"年龄-300岁",也不会对预测结果产生影响。特征离散化的过程，比如特征A，如果当做连续特征使用，在LR模型中，A会对应一个权重w,如果离散化，那么A就拓展为特征A-1，A-2，A-3...,每个特征对应于一个权重，如果训练样本中没有出现特征A-4，那么训练的模型对于A-4就没有权重，如果测试样本中出现特征A-4,该特征A-4也不会起作用。相当于无效。但是，如果使用连续特征，在LR模型中，y = w*a,a是特征，w是a对应的权重,比如a代表年龄，那么a的取值范围是[0..100]，如果测试样本中,出现了一个测试用例，a的取值是300，显然a是异常值，但是w*a还是有值，而且值还非常大，所以，异常值会对最后结果产生非常大的影响。
        - 逻辑回归属于广义线性模型，表达能力受限；单变量离散化为N个后，每个变量有单独的权重，相当于为模型引入了非线性，能够提升模型表达能力，加大拟合；在LR模型中，特征A作为连续特征对应的权重是Wa。A是线性特征，因为y = Wa*A,y对于A的导数就是Wa,如果离散化后，A按区间离散化为A_1,A_2,A_3。那么y = w_1*A_1+w_2*A_2+w_3*A_3.那么y对于A的函数就相当于分段的线性函数，y对于A的导数也随A的取值变动，所以，相当于引入了非线性。
        - 离散化后可以进行特征交叉，加入特征A 离散化为M个值，特征B离散为N个值，那么交叉之后会有M*N个变量，进一步引入非线性，提升表达能力；
        - 特征离散化后，模型会更稳定，比如如果对用户年龄离散化，20-30作为一个区间，不会因为一个用户年龄长了一岁就变成一个完全不同的人。当然处于区间相邻处的样本会刚好相反，所以怎么划分区间是门学问；按区间离散化，划分区间是非常关键的。
        - 特征离散化以后，起到了简化了逻辑回归模型的作用，降低了模型过拟合的风险。(当使用连续特征时，一个特征对应于一个权重，那么，如果这个特征权重较大，模型就会很依赖于这个特征，这个特征的一个微小变化可能会导致最终结果产生很大的变化，这样子的模型很危险，当遇到新样本的时候很可能因为对这个特征过分敏感而得到错误的分类结果，也就是泛化能力差，容易过拟合。而使用离散特征的时候，一个特征变成了多个，权重也变为多个，那么之前连续特征对模型的影响力就被分散弱化了，从而降低了过拟合的风险。)


        模型是使用离散特征还是连续特征，其实是一个“海量离散特征+简单模型” 同 “少量连续特征+复杂模型”的权衡。既可以离散化用线性模型，也可以用连续特征加深度学习。就看是喜欢折腾特征还是折腾模型了。通常来说，前者容易，而且可以n个人一起并行做，有成功经验；后者目前看很赞，能走多远还须拭目以待。
* 标签均衡<br>
  标签比例不平衡（比如1：9），处理方式如下：
    - 上采样（Upsample）（少→多）：<br>
        有放回的重复采样,将少量数据采多
        ```python
        # Separate majority and minority classes
        df_majority = df[df.Survived==0]
        df_minority = df[df.Survived==1]
        
        # Upsample minority class
        df_minority_upsampled = resample(df_minority, 
                                        replace=True,     # sample with replacement
                                        n_samples=(len(df_majority) - len(df_minority)),    # to match majority class
                                        random_state=123) # reproducible results
        
        # Combine majority class with upsampled minority class
        df_upsampled = pd.concat([df_majority, df_minority_upsampled, df_minority])
        print(df_upsampled.columns) 
        print(df_upsampled['Survived'].value_counts())
        ```
    - SMOTE造类似少量数据类别相似的假数据<br>
        参考博客：https://blog.csdn.net/weixin_36431280/article/details/82560988
    - 下采样（Downsample）（多→少）：<br>
        ```python
        Separate majority and minority classes
        df_majority = df[df.Survived==0]
        df_minority = df[df.Survived==1]
        
        # # Upsample minority class
        df_manority_downsampled = resample(df_majority, 
                                        replace=True,     # sample with replacement
                                        n_samples=len(df_minority),    # to match majority class
                                        random_state=123) # reproducible results
        
        # Combine majority class with upsampled minority class
        df_dwonsampled = pd.concat([df_manority_downsampled, df_minority])
        print(df_dwonsampled.columns) 
        print(df_dwonsampled['Survived'].value_counts())
        ```
    - 训练模型控制（推荐：省时、方便）：<br>
        使用模型内部超参（class_weight='balanced'）进行控制，示例如下：<br>
        ```python
        clf = SVC(kernel='linear', 
                  class_weight='balanced', # penalize
                  probability=True)
        ```
    - 特殊情况：<br>
        标签信息太少，将监督学习转为无监督学习的聚类问题进行
## 4.特征选择（过拟合后看看）
* Filter<br>
  比较单特征与标签关系
  - 卡方检验<br>
    经典的卡方检验是检验定性自变量对定性因变量的相关性，参考示例如下：
    ```python
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2#选择K个最好的特征，返回选择特征后的数据
        from sklearn.datasets import load_iris
        iris = load_iris()
        # k Number of top features to select. The “all” option bypasses selection, for use in a parameter search.
        selector = SelectKBest(chi2, k=2).fit(iris.data, iris.target)
        data = selector.transform(iris.data)
        print(data)
        print(selector.scores_)
    ```
    例子：a2_feature_engineering\4_feature_selection\1_filter\chi2.py
  - 方差选择法<br>
    使用方差选择法，先要计算各个特征的方差，然后根据阈值，选择方差大于阈值的特征。使用feature_selection库的VarianceThreshold类来选择特征的代码如下：
    ```python
    from sklearn.feature_selection import VarianceThreshold
    #方差选择法，返回值为特征选择后的数据 #参数threshold为方差的阈值
    from sklearn.datasets import load_iris
    iris = load_iris()
    #print(VarianceThreshold(threshold=3).fit_transform(iris.data))
    print(iris.data[0:5])
    selector = VarianceThreshold(threshold=3).fit(iris.data, iris.target)
    data = selector.transform(iris.data)
    print(data[0:5])
    print(selector.variances_)
    ```
    - 皮尔逊相关系数(Pearson Correlation Coefficient)<br>
      该系数是用来衡量两个数据集合是否在一条线上面，它用来衡量定距变量间的线性关系。<br>
      代码如下：
      ```python
        import numpy as np
        def pcc(X, Y):
        ''' Compute Pearson Correlation Coefficient. '''
        # Normalise X and Y
        X -= X.mean()
        Y -= Y.mean()
        # Standardise X and Y
        X /= X.std()
        Y /= Y.std()
        # Compute mean product
        return np.mean(X*Y)

        # Using it on a random example
        from random import random
        X = np.array([random() for x in range(100)])
        Y = np.array([random() for x in range(100)])
        print(pcc(X, Y))
      ```
* Wrapper<br>
  - 递归特征消除法（也是衡量特征重要性的指标）<br>
    进行大规模实验，进行递归特征消除，返回特征选择后的数据。递归消除特征法使用一个基模型来进行多轮训练，每轮训练后，消除若干权值系数的特征，再基于新的特征集进行下一轮训练。使用feature_selection库的RFE类来选择特征的代码如下：
    ```python
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    iris = load_iris()
    #参数estimator为基模型
    #参数n_features_to_select为选择的特征个数7 
    print(iris.data[0:5])
    selector = RFE(estimator=LogisticRegression(), n_features_to_select=2).fit(iris.data, iris.target)
    data = selector.transform(iris.data)
    print(data[0:5])
    print(selector.ranking_)
    ```
* Embed
    - 线性回归模型：<br>
        通过线性模型学习，比较不同特征的权值，权值越大说明特征更重要，示例如下:
        ```python
        from sklearn.svm import LinearSVC
        from sklearn.datasets import load_iris
        from sklearn.feature_selection import SelectFromModel
        iris = load_iris()
        X, y = iris.data, iris.target
        print(X.shape)
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
        model = SelectFromModel(lsvc, prefit=True)
        X_new = model.transform(X)
        print(X_new.shape)
        ``` 
    - 树模型：<br>
        将特征类别添加至树结构中，距离顶节点越进说明该特征更重要示例如下：
        ```python
        from sklearn.feature_selection import SelectFromModel
        from sklearn.ensemble import GradientBoostingClassifier 
        from sklearn.datasets import load_iris
        iris = load_iris()
        #GBDT作为基模型的特征选择
        #print(SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target))
        selector = SelectFromModel(GradientBoostingClassifier()).fit(iris.data, iris.target)
        print(iris.data[0:5])
        data = selector.transform(iris.data)
        print(data[0:5])
        print(selector.estimator_.feature_importances_)
        ``` 
    > 总结：filter适合特征比较少时使用，通过手工挑选特征；wrapper和embedded适用于自动化处理，用于多特征，但wrapper方式的运算代价远远大于embedded方式，前者需要进行多个模型训练，后者只需训练一个模型。综上所述，推荐使用embedded方式选取特征。 
## 特征扩展（欠拟合后看看）
* 业务总结
    - 根据实际情况整合特征类型：<br>
      - 基本特征<br>
        - 空间：种类、数量、金额、大小、重量、长度
        - 时间：时长，次数，频率，周期 
      - 统计特征<br>
        - 比值、比例
        - 最大值、最小值、平均值、中位数、分位数 
      - 复杂特征<br>
       参考比赛老手打比赛的特征聚合经验
      - 自然特征<br>
      图像、语音、文本、网络 
* 组合已有的特征（组合，聚和)
  - agg函数：
    ```python
    DataFrame.agg（func，axis = 0，* args，** kwargs ）
    ```  
  - gropy函数：<br>
    groupby()是一个分组函数，对数据进行分组操作的过程可以概括为：split-apply-combine三步：<br>
    - 按照键值（key）或者分组变量将数据分组
    - 对于每组应用我们的函数，这一步非常灵活，可以是python自带函数，可以是我们自己编写的函数
    - 将函数计算后的结果聚合<br>

    参考示例：<br>
    ```python
    import pandas as pd
    import numpy as np
    #Create a DataFrame
    d = {
        'Name':['Alisa','Bobby','jodha','jack','raghu','Cathrine',
                'Alisa','Bobby','kumar','Alisa','Alex','Cathrine'],
        'Age':[26,24,23,22,23,24,26,24,22,23,24,24],
        
        'Score':[85,63,55,74,31,77,85,63,42,62,89,77]}
    
    df = pd.DataFrame(d,columns=['Name','Age','Score'])
    print(df.columns.values)
    # key内部求和
    gp = df.groupby(["Name"])["Age"].sum().reset_index() # reset_index重置index
    print(gp)
    gp.rename(columns={"Age":"sum_of_value"},inplace=True) # rename改列名

    print(gp)

    res = pd.merge(df, gp, on=['Name'], how='inner')  # default for how='inner'
    print(res)
    ```
    参考博客：https://blog.csdn.net/songbinxu/article/details/79839363 
  - PolynomialFeatures函数：<br>
    信息利用率低：不同的机器学习算法和模型对数据中信息的利用是不同的，之前提到在线性模型中，使用对定性特征哑编码可以达到非线性的效果。类似地，对定量变量多项式化，或者进行其他的转换，都能达到非线性的效果。<br>
    参考示例如下：
    ```python
    from sklearn.preprocessing import PolynomialFeatures
    import numpy as np
    # a,b -> a, b, ab, a^2, b^2, 1
    X = np.arange(9).reshape(3, 3)
    print(X[0:5])

    poly = PolynomialFeatures(2)
    print(poly.fit_transform(X[0:5]))

    poly = PolynomialFeatures(interaction_only=True)
    print(poly.fit_transform(X))
    ```