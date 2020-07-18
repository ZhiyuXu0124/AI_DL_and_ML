import pickle 

pkl_file = open("data.pkl", "rb") 
datas, label, words_dict, label_dict = pickle.load(pkl_file) 
pkl_file.close()  
print("datas:", datas[:300])
print("label:", label[:300])