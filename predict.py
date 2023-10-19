import os
import pandas as pd
import numpy as np
import json
from scipy.stats import norm
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error

from keras.models import load_model

from mongo_connector import MongoData
from api_wrapper import ApiData
from config import *
from pre_processing import regressor_feature_engineering

from warnings import filterwarnings
filterwarnings('ignore')

class ModelPredictor:
    def __init__(self, model_id=None, active='EURUSD', confidence_level=0.95):
        self.CONFIG = {
            'model_id': model_id,
            'models_folder': r'models'
        }
        self.confidence_level = confidence_level
        self.active = active

        if self.CONFIG['model_id'] is not None:
            print('Inicializando modelo LSTM...')
            self.load_model()
            self.seq_length = self.model_info['seq_length']
            self.CONFIG['new_data_size'] = self.seq_length + 21
            self.mode = 'lstm'

        else:
            print('Inicializando modelo ARIMA...')
            self.seq_length = 20
            self.CONFIG['new_data_size'] = self.seq_length
            self.mode = 'arima'

        # self.get_new_data()

    def load_model(self):
        print(f'Carregando modelo {self.CONFIG["model_id"]}...')
        model_filepath = os.path.join(self.CONFIG['models_folder'], self.CONFIG['model_id'], 'eur_usd.h5')
        model_info_filepath = os.path.join(self.CONFIG['models_folder'], self.CONFIG['model_id'], 'model_info.json')
        
        with open(model_info_filepath, 'r') as file:
            self.model_info = json.load(file)
        self.model = load_model(model_filepath)
        self.scaler_path = os.path.join(self.CONFIG['models_folder'], self.CONFIG['model_id'], 'scaler_lstm.pkl')
        
    def get_new_data(self):
        iq_driver = ApiData(IQ_USER, IQ_PASSWORD)
        mongo = MongoData(active=self.active, ApiData=iq_driver)
        mongo.update()
        new_data = mongo.last_entries(self.CONFIG['new_data_size']).drop(columns={'_id', 'id', 'from', 'to', 'at'}).sort_values(by='from_datetime')
        new_data['from_datetime'] = pd.to_datetime(new_data['from_datetime'])
        new_data.set_index('from_datetime', inplace=True)
        self.new_data = new_data

    def preprocess_data(self):
        self.X_new, self.scaler = regressor_feature_engineering(
            self.new_data,
            seq_length=self.seq_length,
            scaler_path=self.scaler_path,
            train=False,
            columns_order=self.model_info['columns_order']
            )

    def make_predictions(self):
        predictions = self.model.predict(self.X_new) #verbose = 0 para não mostrar o progresso
        
        dummy_df = pd.DataFrame(np.zeros((len(predictions), self.X_new.shape[2])), columns=self.model_info['columns_order'])
        dummy_df['close'] = predictions.ravel()

        # Desnormalizar
        predictions_original_scale_df = pd.DataFrame(self.scaler.inverse_transform(dummy_df), columns=self.model_info['columns_order'])
        self.predictions_original_scale = predictions_original_scale_df['close'].values
        
    def calculate_confidence_interval(self):
        alpha = 1 - self.confidence_level
        z_value = norm.ppf(1 - alpha/2)
        error_mean = self.model_info['mean_error']
        error_std = self.model_info['std_error']
        rmse = self.model_info['rmse']
        error_upper_bound = rmse #error_mean + z_value * error_std
        error_lower_bound = -rmse #error_mean - z_value * error_std

        self.lower_bound = self.predictions_original_scale + error_lower_bound
        self.upper_bound = self.predictions_original_scale + error_upper_bound

    def predict_lstm(self):
        self.preprocess_data()
        self.make_predictions()
        self.calculate_confidence_interval()

    def predict_arima(self, use_confidence_level=True):
        # Verifica se o self.new_data está em uma frequência de 'T' (minutos)
        if self.new_data.index.inferred_freq != 'T':
            self.new_data = self.new_data.asfreq('T')

        seq_data = self.new_data['close'].iloc[-self.seq_length:]

        model = sm.tsa.ARIMA(seq_data, order=(1, 1, 1))
        results = model.fit()

        if use_confidence_level:
            forecast_obj = results.get_forecast(steps=1, alpha=1-self.confidence_level)
            self.predictions_original_scale = np.array([forecast_obj.predicted_mean.iloc[0]])
            self.lower_bound = np.array([forecast_obj.conf_int().iloc[0, 0]])
            self.upper_bound = np.array([forecast_obj.conf_int().iloc[0, 1]])
        else:
            mae = 0.0001
            try:
                mae = self.calculate_mae()
            except:
                pass
            forecast_obj = results.get_forecast(steps=1)
            self.predictions_original_scale = np.array([forecast_obj.predicted_mean.iloc[0]])
            self.lower_bound = self.predictions_original_scale - mae
            self.upper_bound = self.predictions_original_scale + mae

    def calculate_mae(self):
        mongo = MongoData()
        mongo.update_log()

        filter_condition = {
            "close": {"$exists": True},
            "predictor_mode": self.mode
        }
        log_df = pd.DataFrame(list(mongo.log_collection.find(filter_condition)))

        #round predicted price to 6 decimal places
        log_df['predicted_price'] = log_df['predicted_price'].round(6)
        log_df['close'] = log_df['close'].round(6)

        log_df['residual'] = log_df['close'] - log_df['predicted_price']
        mae = mean_absolute_error(log_df['residual'], np.zeros(len(log_df)))
        return mae

    def print_results(self):
        print(self.predictions_original_scale)
        print(f'[{self.lower_bound}, {self.upper_bound}]')

# # Usar a classe
# model_predictor = ModelPredictor(model_id='model_20231003_205958', active='EURUSD')

# model_predictor.get_new_data()
# model_predictor.predict_lstm()
# model_predictor.print_results()