# encoding = utf - 8
'''
Created on 2019年12月25日

@author: lyc
'''
from sklearn.model_selection import train_test_split
import pandas as pd
from bert.bert import syspath

# 读取数据
def load_data():
    content = pd.read_excel(syspath+'/data/data.xlsx',sheetname = 'info')
    neirong = content['内容']
    avg_score = content['avg_score_round']
    labels = []
    for i in range(0,len(neirong)):
        score =0
        if avg_score[i]<40:
            score = -1
        elif avg_score[i]>=40 and avg_score[i]<60:
            score = 0
        else:
            score = 1
        labels.append(score)
    x_train,x_test,y_train,y_test = train_test_split(neirong,labels,test_size = 0.2,random_state=42)
    return x_train,x_test,y_train,y_test

