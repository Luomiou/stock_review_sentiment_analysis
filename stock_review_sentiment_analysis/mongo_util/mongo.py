# encoding = utf - 8
import pymongo
from mongomock.object_id import ObjectId

_host = '127.0.0.1'
_port = 27017
_dbName = 'stock'
_db = None


def get_db():
    global _db
    if _db is None:
        client = pymongo.MongoClient(host=_host, port=_port)
        _db = client[_dbName]
    return _db


def insert(collection, data):
    _db = get_db()
    if type(data) == list:
        # 返回{inserted_id:'5b2369cac315325f3698a1cf'
        return _db[collection].insert_many(data)
    else:
        # 返回{inserted_ids:[ObjectId('5b236aa9c315325f5236bbb6'), ObjectId('5b236aa9c315325f5236bbb7')]}
        return _db[collection].insert_one(data)


def first(collection, query=None, *args, **kwargs):
    _db = get_db()
    return _db[collection].find_one(filter=query, *args, **kwargs)

# db.first('dddd',query={'code':'/sh600001/','title':'aaaa'},projection={'_id':1})


def select(collection, *args, **kwargs):
    _db = get_db()
    return _db[collection].find(*args, **kwargs)


def count(collection, *args, **kwargs):
    _db = get_db()
    return _db[collection].count_documents(*args, **kwargs)


def delete(collection, query):
    _db = get_db()
    # 返回{deleted_count:10}
    return _db[collection].delete_many(query)

def getCollection(collection):
    _db = get_db()
    return _db[collection]

def drop(collection):
    _db = get_db()
    return _db[collection].drop()
def updateOrInsert(collection, update, query=None):
    '''
    修改多条记录。修改满足query的记录所有记录。如果找不到满足query的记录，则insert一条新记录
    '''
    _db = get_db()
    return _db[collection].update_many(filter=query, update=update, upsert=True)


def update_score_class(collection,com,set_value):
    _db = get_db()
    return _db[collection].update(com,set_value)


        
