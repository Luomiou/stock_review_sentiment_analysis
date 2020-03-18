# encoding = utf - 8 

from bert.bert import train_bert,test_bert
from data.read_data import load_data
from xlnet.xlnet import train_xlnet,test_xlnet

def train_many_model():
    print('load data ......')
    x_train, x_test, y_train, y_test = load_data()
    model_name_list = ['Xlnet-CNN','Bert-CNN',
                       'Bert-CNN-3','Xlnet-CNN-3',
                       'Bert-BiLSTM','Xlnet-BiLSTM',
                       'Bert-BiLSTM-CNN-3','Xlnet-BiLSTM-CNN-3',
                       'Bert-BiLSTM-CNN-3-Attention','Xlnet-BiLSTM-CNN-3-Attention']
    # mdoel_name_pict it is generate model's picture, Both XLNET and BERT model are xiangtong
    for model_i in model_name_list:
        print('training ', model_i)
        if 'Xlnet' in model_i:
            model,history = train_xlnet(model_i, x_train, y_train)
            print('trained ', model_i)
            print('evaluation f1 score')
            test_xlnet(x_test,y_test,model,model_i,history)
            del model
        else:
            model,history = train_bert(model_i,x_train,y_train)
            print('trained ',model_i)
            print('evaluation f1 score')
            test_bert(x_test,y_test,model,model_i,history)
            del model
            
# if __name__ == '__main__':
#     train_many_model()