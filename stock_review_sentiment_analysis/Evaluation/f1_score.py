# encoding = utf - 8
'''
Created on 2019年12月25日

@author: lyc
'''

import os
import sys

import keras.backend as k
import numpy as np
import tensorflow as tf 
from Evaluation.f1 import f1

syspath = sys.path[1]
os.chdir(syspath)


# 评估函数f1 score
def Evaluation_f1_score(x,y,model,model_name):
    
    print('training.....')
    x = np.array(x)
    y = np.array(y)
    if 'Bert' in model_name:
        f = open(syspath + '/result/bert/f1.txt', 'a', encoding='UTF-8')
    if 'Xlnet' in model_name:
        f = open(syspath + '/result/xlnet/f1.txt', 'a', encoding='UTF-8')
    # 模型预测
    k.clear_session()
    tf.reset_default_graph()
    # 模型评估
    y = list(y)
    y_pred_ = model.predict(x)
    y_pred_ = list(y_pred_)
    y_pred = []
    for i in y_pred_:
        i = list(i)
        y_pred.append(i.index(max(i)))
    FR_f1, NR_f1, TR_f1, recall,precisio = f1(y_pred, y)
    k.clear_session()
    tf.reset_default_graph()
    print('NR',NR_f1*100)
    print('FR',FR_f1*100)
    print('TR',TR_f1*100)
    f.write('\n')

    f.write(' FR: ' + str(FR_f1) + '\n')
    f.write(' NR: ' + str(NR_f1) + '\n')
    f.write(' TR: ' + str(TR_f1) + '\n')
    f.write(' recall: ' + str(recall) + '\n')
    f.write(' precisio: ' + str(precisio) + '\n')







