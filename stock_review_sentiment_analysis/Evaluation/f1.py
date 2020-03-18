# encoding  = utf - 8

import os
import sys

from cfg.config import eplis
import numpy as np


syspath = sys.path[1]
os.chdir(syspath)


# 计算 f1 值
def f1(y_pre,y_true):
    y_pre = np.array(y_pre)
    y_pre_ = []
    for i in y_pre:
        if i == 2:
            y_pre_.append(-1)
        elif i == 1:
            y_pre_.append(1)
        else:
            y_pre_.append(0)
    FR = 0 # bad -1 看空
    TR = 0 # good 1 看多
    NR = 0 # unknown 0 中性
    for i in range(len(y_pre_)):
        if y_pre_[i] == -1:
            FR += 1
        elif y_pre_[i] == 0:
            NR += 1
        elif y_pre_[i] == 1:
            TR += 1
    FR_count = 0
    NR_count = 0
    TR_count = 0
    y_true_ = []
    for i in y_true:
        i = list(i)
        if i.index(1) == 0:
            y_true_.append(0)
        elif i.index(1) == 1:
            y_true_.append(1)
        else:
            y_true_.append(-1)
    for i in range(len(y_true_)):
        if y_true_[i] == y_pre_[i]:
            if y_true_[i] == -1:
                FR_count += 1
            elif y_true_[i] == 0:
                NR_count += 1
            elif y_true_[i] == 1:
                TR_count += 1
    FR_precision = FR_count / (FR+eplis)
    NR_precisio = NR_count / (NR+eplis)
    TR_precisio = TR_count / (TR+eplis)
    precision = []
    precision.append(FR_precision)
    precision.append(NR_precisio)
    precision.append(TR_precisio)

    FR_T = 0.1
    NR_T = 0.1
    TR_T = 0.1
    for i in range(len(y_true_)):
        if y_true_[i] == -1:
            FR_T += 1
        elif y_true_[i] == 0:
            NR_T += 1
        elif y_true_[i] == 1:
            TR_T += 1
    FR_recall = FR_count / (FR_T + eplis)
    NR_recall = NR_count / (NR_T + eplis)
    TR_recall = TR_count / (TR_T + eplis)
    recall=[]
    recall.append(FR_recall )
    recall.append(NR_recall)
    recall.append(TR_recall)
    FR_f1 = 2 * (FR_precision * FR_recall) / (FR_precision + FR_recall + eplis)
    NR_f1 = 2 * (NR_precisio * NR_recall) / (NR_precisio + NR_recall+ eplis)
    TR_f1 = 2 * (TR_precisio * TR_recall) / (TR_precisio + TR_recall+ eplis)
    return NR_f1,FR_f1,TR_f1,recall,precision
