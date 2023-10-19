from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from joblib import dump
from keras.utils import to_categorical

from predict import ModelPredictor
from pre_processing import regressor_feature_engineering, add_indicators

def add_prediction(df, regressor, df_batch=1000):
    # Preparação para armazenar previsões
    all_predictions = []

    # Calcula o tamanho do lote com base na configuração do regressor
    batch = (regressor.CONFIG['new_data_size']+1) * df_batch

    # Processa o dataframe em lotes
    for i in tqdm(range(0, len(df), batch)):  # O terceiro argumento em range() é o passo (tamanho do lote)
        # Seleciona o lote atual
        batch_data = df.iloc[i:i+batch]

        # Define os novos dados para o regressor e faz as previsões
        regressor.new_data = batch_data
        regressor.X_new, regressor.scaler, index = regressor_feature_engineering(
            regressor.new_data,
            seq_length=regressor.seq_length,
            scaler_path=regressor.scaler_path,
            train=False,
            columns_order=regressor.model_info['columns_order'],
            classifier_train=True
            )

        regressor.make_predictions()

        last_indices = [seq[-1] for seq in index]
        predictions_with_index = list(zip(last_indices, regressor.predictions_original_scale))

        # Armazena as previsões do lote atual
        all_predictions.append(predictions_with_index)

    flattened_predictions = [item for sublist in all_predictions for item in sublist]
    predictions_df = pd.DataFrame(flattened_predictions, columns=['Index', 'Prediction'])
    predictions_df.set_index('Index', inplace=True)
    df_final = df.join(predictions_df, how='left').dropna()

    return df_final

def encode_labels(y):
    y_encoded = y + 1  # Convertendo -1, 0, 1 para 0, 1, 2
    return to_categorical(y_encoded, num_classes=3)

def classifier_feature_engineering(df, regressor_model="model_20231003_205958", scaler_path = None, previous_candles = 20, scaler = None, train=True, columns_order=None):
    regressor = ModelPredictor(regressor_model, active='EURUSD')
    df = add_prediction(df, regressor)
    # df = add_indicators(df)
    df['Daily_Return'] = df['close'].pct_change()
    df['Direction'] = np.where(df['Daily_Return'] > 0, 1, -1)

    # Criar colunas para as direções das candles anteriores
    for i in range(1, previous_candles + 1):
        column_name = f'{i}_Direction'
        df[column_name] = df['Direction'].shift(i)
    df.dropna(inplace=True)

    # Converte todas as colunas criadas
    for i in range(1, previous_candles + 1):
        column_name = f'{i}_Direction'
        df[column_name] = df[column_name].astype(int)

    if train:
        # Crie uma coluna com os timestamps esperados para o fechamento 5 minutos depois
        df['Expected_Timestamp'] = df.index + pd.Timedelta(minutes=5)
        
        # Mapeie esses timestamps para os valores de fechamento
        timestamp_to_close = df['close'].to_dict()
        df['Future_Close'] = df['Expected_Timestamp'].map(timestamp_to_close)

        # Crie a coluna target
        df['target'] = 0
        df.loc[df['close'] < df['Future_Close'], 'target'] = 1
        df.loc[df['close'] > df['Future_Close'], 'target'] = -1

        # Remova as linhas com NaN em 'Future_Close'
        df.dropna(subset=['Future_Close'], inplace=True)

        # Converter a coluna de target para int
        df['target'] = df['target'].astype(int)

        # Limpeza: Remova colunas auxiliares
        df.drop(columns=['Expected_Timestamp', 'Future_Close'], inplace=True)
        
        train_data = df.iloc[:int(0.7 * len(df))]  # Aproximação do tamanho do conjunto de treino

        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(train_data)

        else:
            if columns_order is None:
                raise ValueError("Especifique a ordem das colunas, usada para o scaler")
            df = df[columns_order]
        
        # Salvar o scaler
        dump(scaler, scaler_path)

        # Aplicar transformação em todo o dataframe
        df_normalized = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
        columns_order = df.columns

        y = df_normalized['target'].values
        y = encode_labels(y)

        X = df_normalized.drop(columns=['target']).values

        return X, y, scaler, columns_order