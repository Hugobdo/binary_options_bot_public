import pandas as pd
import os
import datetime
import json
from joblib import load
import dagshub
import mlflow

from keras.models import Sequential, load_model
from keras.layers import Dropout, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping

from mongo_connector import MongoData
from api_wrapper import ApiData
from config import *
from classifier_pre_processing import classifier_feature_engineering

# CONFIGURAÇÕES E HIPERPARÂMETROS
CONFIG = {
    'regressor_model': "model_20231003_205958",
    'checkpoint_filepath': 'models\\checkpoints\\classifier_model_checkpoint.h5',
    'models_folder': 'models\\classifiers',
    'num_previous_candles': 60,
    'batch_size': 256,
    'epochs': 30,
    'early_stopping_patience': 10,
    'dropout_rate': 0.4,
    'dense_units': [100, 75, 50],
    'look_forward': 5,
    'use_model': None,
    'train': True,
    'scaler': None,
    'columns_order': None,
    'rl_initial_threshold': 0.6,
    'rl_learning_rate': 0.01
}

# Criar um identificador único para esta execução
current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_folder_path = os.path.join(CONFIG['models_folder'], f'model_{current_time}')

# Caminho atualizado para salvar o modelo
model_filepath = os.path.join(model_folder_path, 'eur_usd.h5')
CONFIG['model_filepath'] = model_filepath
CONFIG['scaler_filepath'] = os.path.join(model_folder_path, 'scaler_classifier.pkl')

if not os.path.exists(model_folder_path):
    os.makedirs(model_folder_path)

if CONFIG['use_model'] is not None:
    model_info = json.load(open(os.path.join(CONFIG['models_folder'], CONFIG['use_model'], 'model_info.json')))
    CONFIG['columns_order'] = model_info['columns_order']
    CONFIG['scaler'] = load(os.path.join(CONFIG['models_folder'], CONFIG['use_model'], 'scaler_classifier.pkl'))

dagshub.init(DAGSHUB_PROJECT, DAGSHUB_USER, mlflow=True)
iq_driver = ApiData(IQ_USER, IQ_PASSWORD)
mongo = MongoData(active='EURUSD', ApiData=iq_driver)

# CARREGANDO DADOS
print("Carregando dados...")
df = mongo.read_all_data().drop(columns={'_id', 'id', 'from', 'to', 'at'}).sort_values(by='from_datetime')
df['from_datetime'] = pd.to_datetime(df['from_datetime'])
df.set_index('from_datetime', inplace=True)
print("Dados carregados com sucesso!")

print("Criando atributos...")
X, y, scaler, columns_order = classifier_feature_engineering(
    df,
    regressor_model=CONFIG['regressor_model'],
    previous_candles=CONFIG['num_previous_candles'],
    scaler_path=CONFIG['scaler_filepath'],
    scaler=CONFIG['scaler'],
    columns_order=CONFIG['columns_order'],
    train=True
    )

print("Atributos criados com sucesso!")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Dividir os dados em treino, validação e teste
train_size = int(0.7 * len(X))
val_size = int(0.2 * len(X))
test_size = len(X) - train_size - val_size

X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

# Caminho para salvar os checkpoints
# Crie o diretório se ele não existir
checkpoint_dir = os.path.dirname(CONFIG['checkpoint_filepath'])
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Crie os callbacks
checkpoint_callback = ModelCheckpoint(
    filepath=CONFIG['checkpoint_filepath'],
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy', 
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
    print("Modelo treinado carregado com sucesso!")

else:
    # DEFINIÇÃO DO MODELO
    model = Sequential()
    dense_units_list = CONFIG['dense_units']

    # Primeira camada densa com BatchNormalization
    model.add(Dense(dense_units_list[0], activation='relu', input_shape=(X.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dropout(CONFIG['dropout_rate']))

    # Segunda camada densa com BatchNormalization
    model.add(Dense(dense_units_list[1], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(CONFIG['dropout_rate']))

    # Terceira camada densa com BatchNormalization
    model.add(Dense(dense_units_list[2], activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(CONFIG['dropout_rate']))

    # Camada densa para saída
    model.add(Dense(3, activation='softmax'))  # 3 neurônios para as 3 classes

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Modelo criado com sucesso!")

if CONFIG['train']:
    mlflow.tensorflow.autolog()
    with mlflow.start_run():

        history = model.fit(
            X_train, y_train,
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=[checkpoint_callback, early_stopping],
            verbose=1
            )

        # Salve e registre o modelo
        model.save(CONFIG['model_filepath'])
        print("Modelo salvo com sucesso!")
        
        # Apagar o checkpoint após salvar o modelo
        if os.path.exists(CONFIG['checkpoint_filepath']):
            os.remove(CONFIG['checkpoint_filepath'])
            print(f"Checkpoint {CONFIG['checkpoint_filepath']} removido com sucesso!")

    # Testando o modelo
    print("Testando o modelo...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
    print("Teste finalizado com sucesso!")

    # Salvar informações do modelo
    model_info = {
        'Accuracy': test_accuracy,
        'columns_order': columns_order if CONFIG['columns_order'] is None else CONFIG['columns_order'],
        'num_previous_candles': CONFIG['num_previous_candles'],
        'batch_size': CONFIG['batch_size'],
        'epochs': CONFIG['epochs'],
        'early_stopping_patience': CONFIG['early_stopping_patience'],
        'dropout_rate': CONFIG['dropout_rate'],
        'dense_units': CONFIG['dense_units'],
        'look_forward': CONFIG['look_forward']
    }

    with open(os.path.join(model_folder_path, 'model_info.json'), 'w') as f:
        json.dump(model_info, f)