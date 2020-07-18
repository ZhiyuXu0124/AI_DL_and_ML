# 中文文本分词

## 库环境：
- Python: 3.7.0
- TensorFlow ： 1.13.0

## 数据集
训练集选取[data](https://github.com/ZhiyuXu0124/AI_DL_and_ML/tree/master/06_%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E8%AF%8D/data)中的`pku_training.utf8`
## 数据读取及预处理
- 根据分好词的数据建立标签
  - s -> 起始词
  - m -> 中间词
  - e -> 结束词
  - s -> 单独字的词
- [read_data.py](https://github.com/ZhiyuXu0124/AI_DL_and_ML/blob/master/06_%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E8%AF%8D/read_data.py) 
## 模型框架
- 结构：双向RNN
- RNN模块：三层SLTM（隐藏层为128）
- [fenci_model.py](https://github.com/ZhiyuXu0124/AI_DL_and_ML/blob/master/06_%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E8%AF%8D/fenci_model.py)

## 训练阶段
- 训练10000次，批次大小视GPU内存而定（GTX970m：batch_size=128）
- 设置而early stop：最佳准确率达0.98以上且2500次训练后在未提升
- 训练好的模型保存至[model](https://github.com/ZhiyuXu0124/AI_DL_and_ML/tree/master/06_%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E8%AF%8D/model)
- [fenci.py](https://github.com/ZhiyuXu0124/AI_DL_and_ML/blob/master/06_%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E8%AF%8D/fenci.py)

## 测试阶段
- 读取[model/best](https://github.com/ZhiyuXu0124/AI_DL_and_ML/tree/master/06_%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E8%AF%8D/model/best)储存的准确率最高的模型
- 测试语句为：
  - `瓜子二手车直卖网，没有中间商赚差价。车主多卖钱，买家少花钱 `
  - `人们常说生活是一部教科书`
- 结果：
  - `瓜子 二手车 直卖网 ， 没有 中间商 赚 差价 。 车主多 卖 钱 ， 买家 少 花钱`
  - `人们 常说 生活 是 一 部 教科书`
- [test.py](https://github.com/ZhiyuXu0124/AI_DL_and_ML/blob/master/06_%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E8%AF%8D/test.py)