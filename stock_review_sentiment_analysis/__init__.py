# encoding = utf - 8 
from spider.stock_review_spider import spider_data
from train.train_model import train_many_model
from predict.predict import predict_stock_review, get_model


if __name__ == '__main__':
    
    select = [0,1,2,3]
    # 0表示数据采集功能
    # 1表示训练模型功能
    # 2表示股评分类功能
    # 3表示股评情感倾向性识别功能
    if select == 0:
        # 数据采集功能 
        # 以 中国平安 为例
        stockNo = '000001'
        pageNum = 1000
        spider_data(stockNo,pageNum)
    elif select == 1:
        # 训练模型
        # 训练过程 保存了模型结果以及模型结构图
        train_many_model()
    elif select == 2:
        # 股评预测
        # 以中国平安为例
        stockNo = '000001'
        predict_stock_review(stockNo)
    elif select == 3:
        # 股评情感倾向性识别
        string = '还等什么，牛市已经启动了'
        get_model(string)
        
    
        
    


