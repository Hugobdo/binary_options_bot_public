import os
import json
from joblib import load

import pandas as pd
import numpy as np
import dagshub
import mlflow
import datetime
from sklearn.metrics import mean_squared_error

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

from mongo_connector import MongoData
from api_wrapper import ApiData
from config import *
from pre_processing import regressor_feature_engineering, sequence_generator

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f'GPU {gpu} set to memory growth.')
    except RuntimeError as e:
        print(e)

# CONFIGURAÇÕES E HIPERPARÂMETROS
CONFIG = {
    'checkpoint_filepath': r'models/checkpoints/model_checkpoint.h5',
    'models_folder': r'models',
    'seq_length': 60,
    'batch_size': 256,
    'epochs': 30,
    'early_stopping_patience': 10,
    'dropout_rate': 0.4,
    'LSTM_units': [100, 75, 50],
    'use_model': None,
    'scaler': None,
    'columns_order': None,
}

# Criar um identificador único para esta execução
current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_folder_path = os.path.join(CONFIG['models_folder'], f'model_{current_time}')

# Caminho atualizado para salvar o modelo
model_filepath = os.path.join(model_folder_path, 'eur_usd.h5')
CONFIG['model_filepath'] = model_filepath
CONFIG['scaler_filepath'] = os.path.join(model_folder_path, 'scaler_lstm.pkl')
if not os.path.exists(model_folder_path):
    os.makedirs(model_folder_path)

if CONFIG['use_model'] is not None:
    model_info = json.load(open(os.path.join(CONFIG['models_folder'], CONFIG['use_model'], 'model_info.json')))
    CONFIG['columns_order'] = model_info['columns_order']
    CONFIG['scaler'] = load(os.path.join(CONFIG['models_folder'], CONFIG['use_model'], 'scaler_lstm.pkl'))

dagshub.init(DAGSHUB_PROJECT, DAGSHUB_USER, mlflow=True)
iq_driver = ApiData(IQ_USER, IQ_PASSWORD)
mongo = MongoData(active='EURUSD', ApiData=iq_driver)

# CARREGANDO DADOS
print("Carregando dados...")
df = mongo.read_all_data().drop(columns={'_id', 'id', 'from', 'to', 'at'}).sort_values(by='from_datetime')
df['from_datetime'] = pd.to_datetime(df['from_datetime'])
df.set_index('from_datetime', inplace=True)
print("Dados carregados com sucesso!")

print("Criando atributos e sequências...")

X, y, scaler = regressor_feature_engineering(
    df,
    seq_length=CONFIG['seq_length'],
    scaler_path=CONFIG['scaler_filepath'],
    scaler=CONFIG['scaler'],
    train=True,
    columns_order=CONFIG['columns_order']
    )

print("Atributos e sequências criados com sucesso!")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Dividir os dados em treino, validação e teste
train_size = int(0.7 * len(X))
val_size = int(0.2 * len(X))
test_size = len(X) - train_size - val_size

X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

train_generator = sequence_generator(X_train, y_train, CONFIG['batch_size'])
val_generator = sequence_generator(X_val, y_val, CONFIG['batch_size'])

train_steps_per_epoch = len(X_train) // CONFIG['batch_size']
val_steps_per_epoch = len(X_val) // CONFIG['batch_size']

# Caminho para salvar os checkpoints
# Crie o diretório se ele não existir
checkpoint_dir = os.path.dirname(CONFIG['checkpoint_filepath'])
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Crie os callbacks
checkpoint_callback = ModelCheckpoint(
    filepath=CONFIG['checkpoint_filepath'],
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=CONFIG['early_stopping_patience'],
    verbose=1, 
    restore_best_weights=True
    )

# Carrega o checkpoint. Se não, cria um novo modelo.

if os.path.exists(CONFIG['checkpoint_filepath']):
    model = load_model(CONFIG['checkpoint_filepath'])
    print("Checkpoint carregado com sucesso!")

elif CONFIG['use_model'] is not None:
    model = load_model(os.path.join(CONFIG['models_folder'], CONFIG['use_model'], 'eur_usd.h5'))
    print("Modelo carregado com sucesso!")

else:
    # DEFINIÇÃO DO MODELO
    model = Sequential()
    lstm_units_list = CONFIG['LSTM_units']

    # Primeira camada LSTM com BatchNormalization
    model.add(LSTM(
        lstm_units_list[0], 
        activation='tanh', 
        return_sequences=True, 
        input_shape=(X.shape[1], X.shape[2])
        ))
    model.add(BatchNormalization())
    model.add(Dropout(CONFIG['dropout_rate']))

    # Segunda camada LSTM com BatchNormalization
    model.add(LSTM(lstm_units_list[1], 
                   activation='tanh', 
                   return_sequences=True
                   ))
    model.add(BatchNormalization())
    model.add(Dropout(CONFIG['dropout_rate']))

    # Terceira camada LSTM com BatchNormalization
    model.add(LSTM(lstm_units_list[2], 
                   activation='tanh'
                   ))
    model.add(BatchNormalization())
    model.add(Dropout(CONFIG['dropout_rate']))

    # Camada densa para saída
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='huber')
    print("Modelo criado com sucesso!")


mlflow.tensorflow.autolog()
with mlflow.start_run():

    history = model.fit(
        train_generator, 
        validation_data=val_generator,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps_per_epoch,
        epochs=CONFIG['epochs'], 
        callbacks=[checkpoint_callback, early_stopping]
    )

    # Salve e registre o modelo
    model.save(CONFIG['model_filepath'])
    print("Modelo salvo com sucesso!")
    
    # Apagar o checkpoint após salvar o modelo
    if os.path.exists(CONFIG['checkpoint_filepath']):
        os.remove(CONFIG['checkpoint_filepath'])
        print(f"Checkpoint {CONFIG['checkpoint_filepath']} removido com sucesso!")

# Testando RMSE

# Fazer previsões com o modelo treinado
y_pred = model.predict(X_test)

# Criar DataFrames "dummy" para y_test e y_pred
dummy_df_pred = pd.DataFrame(np.zeros((len(y_pred), df.shape[1])), columns=df.columns)
dummy_df_pred['close'] = y_pred.ravel()

dummy_df_test = pd.DataFrame(np.zeros((len(y_test), df.shape[1])), columns=df.columns)
dummy_df_test['close'] = y_test

# Desnormalizar
y_pred_original = pd.DataFrame(scaler.inverse_transform(dummy_df_pred), columns=df.columns)['close'].values
y_test_original = pd.DataFrame(scaler.inverse_transform(dummy_df_test), columns=df.columns)['close'].values

# Calcular RMSE
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
print("RMSE (escala original):", rmse)

errors = y_test_original - y_pred_original
mean_error = np.mean(errors)
std_error = np.std(errors)

model_info = {
    'rmse': rmse,
    'mean_error': mean_error,
    'std_error': std_error,
    'columns_order': df.columns.tolist() if CONFIG['columns_order'] is None else CONFIG['columns_order'],
    'seq_length': CONFIG['seq_length'],
    'batch_size': CONFIG['batch_size'],
    'epochs': CONFIG['epochs'],
    'early_stopping_patience': CONFIG['early_stopping_patience'],
    'dropout_rate': CONFIG['dropout_rate'],
    'LSTM_units': CONFIG['LSTM_units']
}

model_info_filepath = os.path.join(model_folder_path, 'model_info.json')
with open(model_info_filepath, 'w') as file:
    json.dump(model_info, file)

print(f"Informações de erro salvas em {model_info_filepath}")
