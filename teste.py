import pandas as pd
import numpy as np

from mongo_connector import MongoData
from api_wrapper import ApiData
from config import *
from predict import ModelPredictor
from classifier_pre_processing import add_prediction


iq_driver = ApiData(IQ_USER, IQ_PASSWORD)
mongo = MongoData(active='EURUSD', ApiData=iq_driver)

# CARREGANDO DADOS
print("Carregando dados...")
df = mongo.read_all_data().drop(columns={'_id', 'id', 'from', 'to', 'at'}).sort_values(by='from_datetime')
df['from_datetime'] = pd.to_datetime(df['from_datetime'])
df.set_index('from_datetime', inplace=True)
print("Dados carregados com sucesso!")

regressor_model="model_20231003_205958"
regressor = ModelPredictor(regressor_model, active='EURUSD')
df = add_prediction(df, regressor)

df['lower_bound'] = df['Prediction'] - regressor.model_info['rmse']
df['upper_bound'] = df['Prediction'] + regressor.model_info['rmse']
df['close_shift5'] = df['close'].shift(-4)
df['close_shift6'] = df['close'].shift(-5)

# Primeiro, resample o DataFrame para um intervalo de 2 segundos
df_resampled = df.resample('2S').mean()

# Agora, vamos interpolar os valores para 'close', 'max', 'volume', 'open', e 'min'
fields_to_interpolate = ['close', 'max', 'volume', 'open', 'min']
df_resampled[fields_to_interpolate] = df_resampled[fields_to_interpolate].interpolate(method='time')

# Para 'Prediction', 'lower_bound', e 'upper_bound', usaremos ffill para preencher quaisquer buracos
fields_to_ffill = ['Prediction', 'lower_bound', 'upper_bound','close_shift5','close_shift6']
df_resampled[fields_to_ffill] = df_resampled[fields_to_ffill].ffill()

df_resampled['seconds'] = df_resampled.index.second

# Definindo as condições
conditions = [
    (df_resampled['seconds'] < 4) | (df_resampled['seconds'] > 30),
    (df_resampled['close'] < df_resampled['lower_bound']),
    (df_resampled['close'] > df_resampled['upper_bound'])
]

# Definindo as escolhas correspondentes às condições
choices = ["Parado", "Call", "Put"]

# Criando a nova coluna com np.select
df_resampled['action'] = np.select(conditions, choices, default='Pass')

# Você pode optar por excluir a coluna 'seconds' após seu uso
df_resampled = df_resampled.drop('seconds', axis=1)

# Extraia os segundos
seconds = df_resampled.index.second

# Determine com qual coluna comparar baseado nos segundos
df_resampled['compare_close'] = np.where(seconds < 30, df_resampled['close_shift5'], df_resampled['close_shift6'])

# Determine a 'win_action' com base na comparação dos valores de 'close'
df_resampled['win_action'] = np.where(df_resampled['compare_close'] > df_resampled['close'], 'Call', 'Put')

# Limpeza do DataFrame removendo colunas temporárias
df_resampled = df_resampled.drop(['compare_close'], axis=1)


# Filtrar o DataFrame para excluir as linhas onde a ação é 'Parado' ou 'Pass'
df_filtered = df_resampled[(df_resampled['action'] != 'Parado') & (df_resampled['action'] != 'Pass')]

# Calcular acertos e erros
hits = np.sum(df_filtered['action'] == df_filtered['win_action'])
misses = np.sum(df_filtered['action'] != df_filtered['win_action'])

# Calcular a proporção
if hits + misses > 0:  # prevenindo divisão por zero
    hit_ratio = hits / (hits + misses)
else:
    hit_ratio = 0  # ou qualquer valor padrão para casos onde não temos acertos ou erros

print(f"Acertos: {hits}")
print(f"Erros: {misses}")
print(f"Proporção de Acertos: {hit_ratio:.2%}")