# encoding = utf - 8
from lxml import etree
import pymongo
import requests
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
           'AppleWebKit/537.36 (KHTML, like Gecko) '
           'Chrome/67.0.3396.79 Safari/537.36'}

# MongoDB的连接
# client = pymongo.MongoClient('***.***.***.***', 27017)
client = pymongo.MongoClient('*localhost', 27017)
mydb = client['stock']
baseUrl = 'http://guba.eastmoney.com'
# spider
def spider_link(url,stockNo):
    res = requests.get(url,headers) 
    res.encoing = res.apparent_encoding
    selector = etree.HTML(res.text)
    datacoll = mydb[str(stockNo)]
    xpath_x = '//*[@id="articlelistnew"]/div[@class="articleh normal_post odd"]|//*[@id="articlelistnew"]/div[@class="articleh normal_post"]'
    node_list = selector.xpath(xpath_x)
    errorcount = 0
    for each in node_list:
        readcount = each.xpath('./span[1]/text()')[0]
        pingluncount = each.xpath('./span[2]/text()')[0]
        title1 = each.xpath('./span[3]/a/text()')[0]
        titlelink = each.xpath('./span[3]/a/@href')[0]
        reurl = baseUrl+titlelink
        title,content,error,publishTime = get_detail_page_content(reurl)
        error += 1 
        if len(content) == 0 or content == ' ':
            content = title
        if  len(title) == 0:
            content = title1
            content = title1
        errorcount += error
        data_json = {
                    "stockNo":stockNo,
                    "read_count":readcount,
                    "review_count":pingluncount,
                    "title":title,
                    "title_link":reurl,
                    "title_content":content,
                    "publishTime":publishTime,
                    "class":-3,
                    "score":-10
                    }
        datacoll.insert(data_json)
        
    return errorcount

# detail page content
def get_detail_page_content(reurl):
    res = requests.get(reurl,headers) 
    res.encoing = res.apparent_encoding
    selector = etree.HTML(res.text)
    title = ''
    content = ''
    errorcount = 0
    try:
        title = selector.xpath('//*[@id="zwconttbt"]/text()')[0].strip()
        content = selector.xpath('//*[@id="zwconbody"]/div/div/p/text()|//*[@id="zwconbody"]/div/text()')[0].strip()
        datte = selector.xpath('//*[@id="zwconttb"]/div[2]/text()')[0]
        datte = str(datte[4:23])
    except Exception as e:
        print(e)
        errorcount = 1
        print('爬取数据出现错误',reurl)
    return title,content,errorcount,datte


# interface
def spider_data(stockNo,pageNum):
    '''
    @param stockNo:股票代码
    @param pageNum:爬取数据的页数  
    '''
    for i in range(1,pageNum):
        print('正在爬取第',i,'页')
        url = baseUrl+'/list,'+str(stockNo)+'_'+str(i)+'.html'
        errorcount = spider_link(url,stockNo)
        print('第',i,'页,有',errorcount,'个数据没有爬取到')
    
    