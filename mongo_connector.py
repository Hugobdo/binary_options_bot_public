from tqdm import tqdm
import pymongo
import time
import pandas as pd
import json
import numpy
from datetime import datetime

from config import *

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(CustomEncoder, self).default(obj)


class MongoData:
    def __init__(self, active='EURUSD', ApiData=None):
        self.client = CLIENT
        self.db = self.client.binary_bot
        self.active = active
        self.active_collection = self.db[self.active]
        self.log_collection = self.db['operations']
        self.apidata = ApiData
        self.first_date = None
        self.last_date = None

    def insert_historical(self, rows):
        if self.apidata is None:
            raise ValueError('An Api connection is required to update the database!')
        try:
            self.first_date = list(self.active_collection.find().sort("from", pymongo.ASCENDING).limit(1))[0]["from"]
            end_date = self.first_date
        except IndexError:
            current_time = datetime.fromtimestamp(time.time())
            end_date = current_time.replace(second=0, microsecond=0).timestamp()-60
        for i in tqdm(range(int(rows / 1000))):
            data = self.apidata.api.get_candles(self.active, 60, 1000, end_date)
            for candle in data:
                candle['from_datetime'] = datetime.fromtimestamp(candle['from'])
            self.active_collection.insert_many(data)
            end_date = int(data[0]["from"]) - 1

    def update(self):
        if self.apidata is None:
            raise ValueError('An Api connection is required to update the database!')
        
        print('Updating database...')
        # Pega o tempo atual e ajusta para o último minuto completo
        current_time = datetime.fromtimestamp(time.time())
        end_date = current_time.replace(second=0, microsecond=0).timestamp()-60

        self.last_date = list(self.active_collection.find().sort("from", pymongo.DESCENDING).limit(1))[0]["from"]
        while end_date > self.last_date:
            data = self.apidata.api.get_candles(self.active, 60, 1, end_date)
            if data[-1]["from"] == self.last_date:
                break
            for candle in data:
                candle["from_datetime"] = datetime.fromtimestamp(candle["from"])
            self.active_collection.insert_many(data)
            end_date = int(data[0]["from"]) - 1
            print(f'Inserted {data[0]["from_datetime"]} on database')
        print('Database updated!')

    def read_all_data(self):
        return pd.DataFrame.from_records(self.active_collection.find())

    def last_entries(self, n=100):
        return pd.DataFrame.from_records(self.active_collection.find().sort("from", pymongo.DESCENDING).limit(n))

    def update_log(self, verbose=False):
        '''
        If the close price is missing on the log, update it
        '''
        documents = self.log_collection.find({"close": {"$exists": False}})
        for document in documents:
            close_time = document['end_time']
            self.active_collection = self.db[document['active']]
            candle = self.active_collection.find_one({"from_datetime": close_time})
            if candle is None or candle["from_datetime"] != close_time:
                if verbose:
                    print(f"Could not find candle for {close_time}")
                continue
            self.log_collection.update_one({"_id": document["_id"]}, {"$set": {"close": candle["close"]}})


# from api_wrapper import ApiData
# iq_driver = ApiData(IQ_USER, IQ_PASSWORD)
# mongo = MongoData(active='EURUSD', ApiData=iq_driver)
# mongo.insert_historical(10000000)

