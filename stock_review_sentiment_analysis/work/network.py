# encoding = utf - 8
'''
Created on 2019年12月20日

@author: lyc
'''
import sys

from keras import optimizers, losses, Input
from keras.activations import softmax, relu
from keras.engine.training import Model
from keras.layers.convolutional import Conv1D
from keras.layers.core import Permute, Dropout, Dense
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from matplotlib import pyplot as plt

from cfg.config import learning_rate, keep_prob, input_length, epochs, batch_size, units, \
     class_nums, kernel_size
from Att.attention_keras import AttentionLayer
import yaml


syspath = sys.path[1]

def design_network(model_name):
    input1 = Input(shape=(input_length,768))
    if model_name == 'CNN':
        lcov1=Conv1D(filters=units, kernel_size=1, activation=relu)(input1)
        out = MaxPooling1D(pool_size=1)(lcov1)
    if model_name == 'BiLSTM':
        out = Bidirectional(LSTM(units))(input1)
    if 'Bert' in model_name or 'Xlnet' in model_name:
        convs = []
        for fsz in kernel_size:
            l_conv = Conv1D(filters=units, kernel_size=fsz, activation=relu)(input1)
            lpool = MaxPooling1D(input_length - fsz + 1)(l_conv)
            convs.append(lpool)
        merge = concatenate(convs, axis=1)
            
    #   reshape = Reshape((units,3))(merge)
        permute = Permute((2,1))(merge)
        if 'Att' in model_name:
            out = Bidirectional(LSTM(units,return_sequences=True))(permute)
            out = AttentionLayer(step_dim=units)(out)
        else:
            out = Bidirectional(LSTM(units))(permute)
    out = Dropout(keep_prob)(out)
    output = Dense(class_nums, activation=softmax)(out)
    model = Model(input1, output)
    model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adam(lr=learning_rate),
                          metrics=['accuracy'])
    model.summary()
    return model

def xlblc3a():
    input1 = Input(shape=(input_length,768))
    biout = Bidirectional(LSTM(units,return_sequences=True))(input1)
    convs = []
    for fsz in kernel_size:
        l_conv = Conv1D(filters=units, kernel_size=fsz, activation=relu)(biout)
        lpool = MaxPooling1D(input_length - fsz + 1)(l_conv)
        convs.append(lpool)
    merge = concatenate(convs, axis=1)
    out = AttentionLayer(step_dim=units)(merge)
    out = Dropout(keep_prob)(out)
    output = Dense(class_nums, activation=softmax)(out)
    model = Model(input1, output)
    model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adam(lr=learning_rate),
                          metrics=['accuracy'])
    model.summary()

# save model picture
def save_model_picture(model,model_name,x_train,y_train):
    print('training')
    history = model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,shuffle=True)
    history = history.history
    loss = history['loss']
    accuracy = history['acc']
    plt.plot()
    plt.plot(range(epochs),loss,'o-')
    plt.plot(range(epochs),accuracy,'.-')
    plt.ylabel('Train accuracy')
    plt.legend(['loss', 'accuracy'], loc='lower left')
    plt.savefig(syspath+'/model_picture/'+model_name)
    
# save model
def save_model(model,model_name):
    yaml_string = model.to_yaml()
    with open(syspath+'/model_picture/'+model_name+'.yml','w') as f:
        f.write(yaml.dump(yaml_string,default_flow_style=True))
    model.save_weights(syspath+'/model_picture/'+model_name+'.h5')






