# encoding = utf - 8

'''
Created on 2019/12/20

@author: lyc
'''
import codecs
import os
import sys

import keras
from keras_bert.loader import load_trained_model_from_checkpoint
from keras_bert.tokenizer import Tokenizer

from cfg.config import input_length, class_nums
from work.network import design_network, save_model_picture, save_model
from Evaluation.f1_score import Evaluation_f1_score

syspath = sys.path[1]
os.chdir(syspath)

config_path = syspath+'/bert/chinese_L-12_H-768_A-12/bert_config.json'# ���������ļ�
checkpoint_path = syspath+'/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = syspath+'/bert/chinese_L-12_H-768_A-12/vocab.txt'

# class stock_review:
#     def __init__ (self,title,content,score):
#         self.title = title
#         self.content = content
#         self.score = score

# def load_data():
#     
#     content = pd.read_excel(syspath+'/data/data.xlsx',sheetname = 'info')
#     title = content['标题']
#     neirong = content['内容']
#     avg_score = content['avg_score_round']
#     stock_list = []
#     labels = []
#     for i in range(0,len(title)):
#         score =0
#         if avg_score[i]<40:
#             score = -1
#         elif avg_score[i]>=40 and avg_score[i]<60:
#             score = 0
#         else:
#             score = 1
#         stock = stock_review(str(title[i]),str(neirong[i]),score)
#         stock_list.append(stock)
#         labels.append(score)
#     return stock_list,labels
# encode
def get_token_dict(dict_path):
    '''
    :param: dict_path: 是bert模型的vocab.txt文件
    :return:将文件中字进行编码
    '''
    # 将bert模型中的 字 进行编码
    # 目的是 喂入模型  的是  这些编码，不是汉字
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf-8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    tokenizer = Tokenizer(token_dict)
    return tokenizer  


# 得到编码

def get_encode(stock):
    tokenizer = get_token_dict(dict_path)
    X1 = []
    X2 = []
    for line in stock:
#         x1,x2 = tokenizer.encode(first=line.title,second=line.content,max_len=input_length)
        x1,x2 = tokenizer.encode(first=str(line),max_len=input_length)
        X1.append(x1)
        X2.append(x2)
    return X1,X2
def build_bert_model(X1,X2):
    bert_model = load_trained_model_from_checkpoint(config_path,checkpoint_path,seq_len=input_length)
    wordvec = bert_model.predict([X1, X2])
    return wordvec
 
# train
def train_bert(model_name,x_train,y_train):
    print('get encode .....')
    X1,X2 = get_encode(x_train)
    labels = keras.utils.to_categorical(y_train, num_classes=class_nums)
    print('predicting word vector ......')
    word2vec = build_bert_model(X1,X2)
    print(word2vec.shape)
    print('loading network......')
    model = design_network(model_name)
    print('saving model picture......')
    save_model_picture(model,model_name,word2vec,labels)
    print('saving model .....')
    save_model(model, model_name)
    return model


# test bert
def test_bert(x_test,y_test,model,model_name):
    X1,X2 = get_encode(x_test)
    labels = keras.utils.to_categorical(y_test, num_classes=class_nums)
    print('predicting word vector ......')
    word2vec = build_bert_model(X1,X2)
    Evaluation_f1_score(word2vec,labels,model,model_name)
    