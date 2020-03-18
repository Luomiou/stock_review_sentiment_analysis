# encoding = utf - 8
'''
Created on 2019年12月24日

@author: lyc
'''
import os
import sys

from matplotlib.font_manager import FontProperties

import matplotlib.pyplot as plt
import pandas as pd


font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  
syspath = sys.path[1]
os.chdir(syspath)


# 加载数据
def load_data():
    
    content = pd.read_excel(syspath+'/data/data.xlsx',sheetname = 'info')
    neirong = content['内容']
        
    x = [10,20,30,40,50,60,70,80,90,100,110]
    y = []
    one = 0
    ten = 0
    twoty = 0
    threety = 0
    forty = 0
    fivty = 0
    sixty = 0
    seventy = 0
    eity = 0
    ninty = 0
    hu = 0
    for i in neirong:
        strlength = len(str(i))
        if strlength >= 100:
            hu += 1
        if strlength >= 90 and strlength < 100:
            ninty += 1   
        if strlength >= 80 and strlength < 90:
            eity += 1
        if strlength >= 70 and strlength < 80:
            seventy += 1
        if strlength >= 60 and strlength < 70:
            sixty += 1
        if strlength >= 50 and strlength < 60:
            fivty += 1
        if strlength >= 40 and strlength < 50:
            forty += 1 
        if strlength >= 30 and strlength < 40:
            threety += 1 
        if strlength >= 20 and strlength < 30:
            twoty += 1 
        if strlength >= 10 and strlength < 20:
            ten += 1 
        if strlength < 10:
            one += 1 
    y.append(one)
    y.append(twoty)
    y.append(ten)
    y.append(threety)
    y.append(forty)
    y.append(fivty)
    y.append(sixty)
    y.append(seventy)
    y.append(eity)
    y.append(ninty)
    y.append(hu)
    plt.bar(x, y, label=u'sentence length frequent')
    plt.legend()
    plt.xlabel(u'股评长度',FontProperties = font)
    plt.ylabel(u'频数',FontProperties = font)
    plt.title(u'股评长度统计', FontProperties=font)
    plt.show()
load_data()


