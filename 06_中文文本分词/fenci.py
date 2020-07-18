import time
from datetime import timedelta

import numpy as np

from fenci_model import Model
from read_data import Data
from utils import _get_label, get_time_dif


def main():
    batch_size = 128 
    seq_len = 50
    lr = 0.001
    n_layer=2
    hidden_size=64
    epochs = 10000               # 训练次数
    total_batch = 0              # 总批次
    best_acc = 0.0               # 最佳验证集准确率
    last_improved = 0            # 记录上一次提升批次
    require_improvement = 2500   # 如果超过2000轮未提升，提前结束训练
    data_tool = Data(read_path='data/pku_training.utf8', save_path='data/') 
    model = Model(batch_size=None, is_training=True, n_layer=n_layer, hidden_size=hidden_size, n_words=data_tool.n_words, learning_rate=lr) 
    model.init_sess("model")


    print('***************Training***************')
    start_time = time.time()
    for step in range(epochs):
        data, label = data_tool.batch_data(batch_size, seq_len)
        loss, acc = model.train(np.array(data), np.array(label), np.ones([batch_size, seq_len]), np.ones([batch_size])*seq_len)
        if step%50==0:
            model.save(model="model/rnn_3", itr=step)
            times = get_time_dif(start_time)
            if acc >= best_acc:
                best_acc = acc
                last_improved = step
                improved_str = '*' 
                model.save(model="model/best/rnn_3", itr=step)
            else:
                improved_str = ''
            print(f'Steps:{step}, Loss:{loss}, Accuarcy:{acc*100}%, Time:{times}{improved_str}')

        total_batch += 1
        if ((total_batch - last_improved) > require_improvement) and (best_acc >= 0.98):
            # 训练集正确率长期不提升，提前结束训练
            print("No optimization for a long time, auto-stopping...")
            break  # 跳出循环      
    
    print('***************Testing***************')
    words = ["瓜子二手车直卖网，没有中间商赚差价。车主多卖钱，买家少花钱", "人们常说生活是一部教科书"]
    for word in words:
        word_id = data_tool.get_id(word)
        idx = model.valid(word_id) 
        lb = data_tool.seg_words(word, idx)
        print(lb)

            
if __name__=="__main__":
    main()
