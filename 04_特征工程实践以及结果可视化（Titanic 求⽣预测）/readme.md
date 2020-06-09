# Titanic求生预测
- 特征工程数据处理
- 结果可视化

## 项目环境
- Python: 3.7 
- pandas: 0.23.4
- matplotlib: 2.2.3
- numpy: 1.16.4
- sklearn: 0.22.2.post1
  
## 数据介绍
各列信息：

| 列名 | 属性  |
| ------------- |:-------------:| 
| "row.names" |  ID| 
| "pclass"  |   类别| 
| "survived"  |  是否获救| 
| "name"  |    名字| 
| "age"   |    年龄| 
| "embarked"  |  从事工作| 
| "home.dest" |   家乡地址| 
| "room"    |     房间| 
| "ticket"  |   船票| 
| "boat"  |     船号| 
| "sex"  |     性别| 

## 数据预处理
- 填充缺失值
- 特征离散/因子化
- 无量纲化

## 算法
- KNN
- 随机森林
- 决策树
>KNN表现较好