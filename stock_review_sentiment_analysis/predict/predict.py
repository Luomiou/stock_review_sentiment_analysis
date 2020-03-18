# encoding = utf - 8
import sys

from keras import backend as k, losses, optimizers
from keras.models import model_from_yaml
import yaml

from Att.attention_keras import AttentionLayer
from cfg.config import learning_rate,aerfa
from xlnet.xlnet import  config_path, checkpoint_path,\
    dict_path
import numpy as np
from keras_xlnet.loader import load_trained_model_from_checkpoint
from keras_xlnet.tokenizer import Tokenizer
import tensorflow as tf
import gc
from mongo_util import mongo as db

syspath = sys.path[1]
# 得到编码
def build_xlnet_model(texts):
    xlnet_model =  load_trained_model_from_checkpoint(
                    config_path,
                    checkpoint_path,
                    batch_size = 1,
                    target_len = 30,
                    memory_len = 0,
                    in_train_phase = False
                    )
    tokenizer = Tokenizer(dict_path)

    tokens = tokenizer.encode(str(texts))
    tokens = tokens+[0]*(30-len(tokens))if len(tokens)<30 else tokens[0:30]
    token_input = np.expand_dims(np.array(tokens), axis=0)
    segment_input = np.zeros_like(token_input)
    memory_length_input = np.zeros((1,1))
    result = xlnet_model.predict([token_input,segment_input,memory_length_input])
    results = np.array(result)
    del xlnet_model
    del tokenizer
    del tokens
    del token_input
    del segment_input
    del memory_length_input
    del result
    return results


def get_model(string):
    k.clear_session()
    print('loading model yaml')
    with open(syspath+'/model_picture/Xlnet-BiLSTM-CNN-3-Attention.yml','r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string,custom_objects={'AttentionLayer':AttentionLayer})
    print('loading model weights')
    model.load_weights(syspath+'/model_picture/XLnet-BiLSTM-CNN-3-Attention.h5','r')
    model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adam(lr=learning_rate),
                          metrics=['accuracy'])
    try:
        result = model.predict(string)
        result = np.array(result)
        print(result)
        x = np.argmax(result)
        print(x)
    except Exception as e:
        print(e)
    
    del model
    del result
    gc.collect()
    return x


def generator_data(collectionName):
    '''
    :param collectionName:数据库名称
    :return:
    '''
    data = db.select(collectionName, {'$and': [{'class': -3}, {'score': -10}]})
    for i in data:
        yield i
        
        
def cal_score(review_count,read_count):
    if '.'not in review_count:
        reviewshu = float(review_count)
    else:
        index = review_count.index("万")
        reviewshu = float(review_count[:index])*10000+float(review_count[index:])*100
    if '.'not in read_count:
        readshu = float(read_count)
    else:
        index = read_count.index("万")
        readshu = float(read_count[:index])*10000+float(read_count[index:])*100
    score = readshu * aerfa + reviewshu * (1 - aerfa)
    del review_count
    del reviewshu
    del read_count
    del readshu
    print(score)
    return score


def predict_stock_review(collectionName):
    for c in generator_data(collectionName=collectionName):
        id1 = c['_id']
        content = c['title_content']
        print('content',content)
        read_count = c['read_count']
        review_count = c['review_count']
        score = cal_score(review_count, read_count)
        
        content_vector = build_xlnet_model(content)
        print('vector')
        content_vector = np.array(content_vector)
        print(content_vector.shape)
        label = get_model(content_vector)
        if label == 2:
            label = -1
        db.update_score_class(collectionName, com={'_id':id1}, set_value = {'$set':{'score':int(score),'class':int(label)}})
    tf.reset_default_graph()




