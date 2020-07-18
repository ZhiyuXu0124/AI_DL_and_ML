from read_data import Data
from fenci_model import Model

batch_size = 128 
seq_len = 50
lr = 0.001
n_layer=2
hidden_size=64
data_tool = Data(read_path='data/pku_training.utf8', save_path='data/')
model = Model(batch_size=None, is_training=True, n_layer=n_layer, hidden_size=hidden_size, n_words=data_tool.n_words, learning_rate=lr) 
model.init_sess('model')
words = ["瓜子二手车直卖网，没有中间商赚差价。车主多卖钱，买家少花钱", "人们常说生活是一部教科书"]
for word in words:
    word_id = data_tool.get_id(word)
    idx = model.valid(word_id) 
    lb = data_tool.seg_words(word, idx)
    print(lb)