from pymongo import MongoClient
import os

# IQ Option Config
IQ_USER = 'YOUR_IQ_OPTION_USER'
IQ_PASSWORD ='YOUR_IQ_OPTION_PASSWORD'

# MLFlow Config - Dagshub
DAGSHUB_USER = 'YOUR_DAGSHUB_USER'
DAGSHUB_PROJECT = 'YOUR_DAGSHUB_PROJECT'
os.environ['MLFLOW_TRACKING_URI'] = f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_PROJECT}.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "yourUsername"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "yourDagshubToken"

#Your MongoDB Connection string. If using localhost, leave as is
CLIENT = MongoClient('mongodb://localhost')
