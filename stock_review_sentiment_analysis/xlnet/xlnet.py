# encoding = utf - 8

'''
Created on 2019/12/20

@author: lyc
'''
import os
import sys

import keras
from keras_xlnet.loader import load_trained_model_from_checkpoint
from keras_xlnet.tokenizer import Tokenizer

from cfg.config import input_length, class_nums, batch_size
from work.network import design_network, save_model_picture, save_model
import numpy as np
from Evaluation.f1_score import Evaluation_f1_score

syspath = sys.path[1]
os.chdir(syspath)

config_path = syspath+'/xlnet/chinese_xlnet_mid_L-24_H-768_A-12/xlnet_config.json'
checkpoint_path = syspath+'/xlnet/chinese_xlnet_mid_L-24_H-768_A-12/xlnet_model.ckpt'
dict_path = syspath+'/xlnet/chinese_xlnet_mid_L-24_H-768_A-12/spiece.model'


# 得到编码
def build_xlnet_model(texts):
    xlnet_model =  load_trained_model_from_checkpoint(
                    config_path,
                    checkpoint_path,
                    batch_size = batch_size,
                    target_len = input_length,
                    memory_len = 1,
                    in_train_phase = False
                    )
    tokenizer = Tokenizer(dict_path)
    results = []
    for text in texts:
        tokens = tokenizer.encode(str(text))
        tokens = tokens+[0]*(input_length-len(tokens))if len(tokens)<input_length else tokens[0:input_length]
        token_input = np.expand_dims(np.array(tokens), axis=0)
        segment_input = np.zeros_like(token_input)
        memory_length_input = np.zeros((1,1))
        result = xlnet_model.predict([token_input,segment_input,memory_length_input])
#         print(result.shape)
        results.append(result)
    return results

 
# train
def train_xlnet(model_name,x_train,y_train):
    labels = keras.utils.to_categorical(y_train, num_classes=class_nums)
    print('\n')
    print('predicting word vector ......')
    results = build_xlnet_model(x_train)
    print('loading network......')
    model = design_network(model_name)
    print('saving model picture......')
    save_model_picture(model,model_name,results,labels)
    print('saving model .....')
    save_model(model, model_name)
    
# test xlnet
def test_xlnet(x_test,y_test,model,model_name):
    labels = keras.utils.to_categorical(y_test, num_classes=class_nums)
    print('\n')
    print('predicting word vector ......')
    results = build_xlnet_model(x_test)
    print(results.shape)
    print('predicting word vector ......')
    word2vec = build_xlnet_model(x_test)
    Evaluation_f1_score(word2vec,labels,model,model_name)
