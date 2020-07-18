import os
import pickle
import numpy as np
from utils import _get_label

class Data():
    def __init__(self, read_path, save_path):
        self.read_path=read_path
        self.save_path=save_path
        if os.path.exists(self.save_path+"data.pkl")==False:
            files = open(self.read_path, "r", encoding="utf-8")
            file_data = files.read().replace("\n", "  ") 
            files.close()
            file_data_set = set(file_data)
            words_dict = dict(zip(file_data_set, range(len(file_data_set))))
            label_dict = {"b":0, "m":1, "s":2, "e":3}
            file_data_seg = [itr for itr in file_data.split(" ") if len(itr)>0] 
            file_data_label = [_get_label(itr) for itr in file_data_seg] 
            datas = "".join(file_data_seg) 
            label = "".join(file_data_label) 
            pkl_file = open(self.save_path+"data.pkl", "wb") 
            pickle.dump([datas, label, words_dict, label_dict], pkl_file) 
            pkl_file.close() 
        else:
            pkl_file = open(self.save_path+"data.pkl", "rb") 
            datas, label, words_dict, label_dict = pickle.load(pkl_file) 
            pkl_file.close()             
        self.words_dict = words_dict 
        self.label_dict = label_dict 
        self.datas = [self.words_dict[itr] for itr in datas] 
        self.label = [self.label_dict[itr] for itr in label] 
        self.index = 0 
        self.length = len(self.datas)
        self.n_words = len(self.words_dict)
    def batch_data(self, batch_size=32, length=50):
        datas = [] 
        label = []
        for _ in range(batch_size):
            datas.append(self.datas[self.index:self.index+length])
            label.append(self.label[self.index:self.index+length]) 
            if self.index+2*length>=self.length:
                self.index = 0
            else:
                self.index += length
        return datas, label
    def batch_iter(self, batch_size=64):
        """生成批次数据"""
        data_len = len(self.datas)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = self.datas[indices]
        y_shuffle = self.label[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id] 
    def get_id(self, words):
        return [self.words_dict[itr] for itr in words]
    def get_label(self, labelid):
        lbdic = {0:"b", 1:"m", 2:"s", 3:"e"} 
        return "".join([lbdic[itr] for itr in labelid]) 
    def seg_words(self, words, labelid):
        wrd = []
        for w, ids in zip(words, labelid):
            if ids==2 or ids==3:
                wrd.append(w)
                wrd.append(" ") 
            else:
                wrd.append(w) 
        return "".join(wrd)