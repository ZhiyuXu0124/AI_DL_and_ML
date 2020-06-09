#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
# @file     run.py
# @author   Zhiyu Xu <ZHiyu.Xu19@student.xjtlu.edu.cn>
# @date     2020-06-9
#
# @brief    Code fot Click-Through Rate Prediction(https://www.kaggle.com/c/avazu-ctr-prediction/overview/description),
#           Only part of the data is selected for running speed.
#

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler


# read data
data_train = pd.read_csv(r'data\train_sample_ctr_8450.csv')
data_test = pd.read_csv(r'data\test_sample_ctr_155.csv')

# drop 'id' feature
data_train = data_train.drop('id',axis=1)
data_test = data_test.drop('id',axis=1)


site_id = pd.get_dummies(data_train['site_id'], prefix= 'site_id')
site_domain = pd.get_dummies(data_train['site_domain'], prefix= 'site_domain')
site_category = pd.get_dummies(data_train['site_category'], prefix= 'site_category')
app_id = pd.get_dummies(data_train['app_id'], prefix= 'app_id')
app_domain = pd.get_dummies(data_train['app_domain'], prefix= 'app_domain')
app_category = pd.get_dummies(data_train['app_category'], prefix= 'app_category')
device_id = pd.get_dummies(data_train['device_id'], prefix= 'device_id')
device_ip = pd.get_dummies(data_train['device_ip'], prefix= 'device_ip')
device_model = pd.get_dummies(data_train['device_model'], prefix= 'device_model')

df = pd.concat([site_id, site_domain, site_category, 
                app_id, app_domain, app_category, 
                device_id, device_ip, device_model], axis=1)


scaler = MinMaxScaler()
temp = scaler.fit_transform(data_train.filter(regex='hour|C.*|banner_pos|device_type|device_conn_type'))
trian_data_scalered = pd.DataFrame(temp, 
                    columns=['hour_s','C1_s','C14_s','C15_s','C16_s','C17_s','C18_s',
                            'C19_s','C20_s','C21_s','banner_pos_s','device_type_s','device_conn_type_s'])
selected_feature_df = pd.concat([data_train['click'], trian_data_scalered,site_domain,site_category,
                                 app_domain,app_category,device_model], axis=1)
# train data
X = selected_feature_df.values[:,1:]
# train target
y = selected_feature_df.values[:,0]

clf = linear_model.LogisticRegression(C=1.0, penalty='l2', class_weight='balanced', n_jobs=-1, solver='lbfgs')
scores = cross_val_score(clf, X, y, cv=5)
print('accuracy:{:.3f}%'.format(np.mean(scores)*100))