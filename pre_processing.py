import os

import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tqdm import tqdm

def create_sequences_train(data, seq_length):
    xs = []
    ys = []

    for i in tqdm(range(len(data) - seq_length - 4)):
        current_sequence = data.iloc[i:(i+seq_length)]
        
        # Verifica se os timestamps são consecutivos
        time_diff = current_sequence.index.to_series().diff().dropna()
        if not (time_diff == pd.Timedelta(minutes=1)).all():
            continue
        
        x = current_sequence.values
        y = data.iloc[i+seq_length+4]['close']
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

def create_sequences_predict(data, seq_length, classifier_train=False):
    xs = []
    indexes = []

    for i in range(len(data) - seq_length):
        # Selecionando a sequência atual
        current_sequence = data.iloc[i:(i + seq_length)]
        x = current_sequence.values
        xs.append(x)

        if classifier_train:
            index = current_sequence.index
            indexes.append(index)

    if classifier_train:
        return np.array(xs), indexes
    else:
        return np.array(xs)

def add_indicators(df):
     # FEATURE ENGINEERING
    df['Hour'] = df.index.hour
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['MA_21'] = df['close'].rolling(21).mean()
    df['Momentum'] = df['close'] - df['close'].shift(1)

    # Adicionando MACD
    EMA_12 = df['close'].ewm(span=12).mean()
    EMA_26 = df['close'].ewm(span=26).mean()
    df['MACD'] = EMA_12 - EMA_26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()

    # Adicionando RSIx
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Adicionando ATR
    df['HL'] = df['max'] - df['min']
    df['HC'] = (df['max'] - df['close'].shift(1)).abs()
    df['LC'] = (df['min'] - df['close'].shift(1)).abs()
    df['TR'] = df[['HL', 'HC', 'LC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    # Adicionando Stochastic Oscillator (%K)
    low_min = df['min'].rolling(14).min()
    high_max = df['max'].rolling(14).max()
    df['%K'] = (df['close'] - low_min) * 100 / (high_max - low_min)

    # MACD Histogram
    df['MACD_Histogram'] = df['MACD'] - df['MACD_signal']

    # Bollinger Bands
    df['MA_20'] = df['close'].rolling(20).mean()
    df['BB_upper'] = df['MA_20'] + 2 * df['close'].rolling(20).std()
    df['BB_lower'] = df['MA_20'] - 2 * df['close'].rolling(20).std()

    # OBV
    df['Daily_Return'] = df['close'].pct_change()
    df['Direction'] = np.where(df['Daily_Return'] > 0, 1, -1)
    df.loc[df['Daily_Return'] == 0, 'Direction'] = 0  # Neutral days
    df['Volume_Direction'] = df['Direction'] * df['volume']
    df['OBV'] = df['Volume_Direction'].cumsum()
   
    # Support and Resistance using rolling maximum and minimum
    window = 20  # Você pode alterar a janela conforme sua necessidade
    df['Resistance'] = df['max'].rolling(window=window).max()
    df['Support'] = df['min'].rolling(window=window).min()

    # Adicionando retardos do Momentum
    for i in range(1, 4):
        df[f'Momentum_Lag_{i}'] = df['Momentum'].shift(i)

    df.drop(columns=['HL', 'HC', 'LC', 'TR'], inplace=True)  # Removendo colunas intermediárias
    df.dropna(inplace=True)
    return df

def regressor_feature_engineering(df, seq_length = None, scaler_path = None, scaler = None, train=True, columns_order=None, classifier_train=False):

    df = add_indicators(df)
    if train:
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

        X, y = create_sequences_train(df_normalized, seq_length)
        return X, y, scaler

    else:

        if columns_order is None:
            raise ValueError("Especifique a ordem das colunas")

        if scaler is None:
            scaler = load(scaler_path)

        df = df[columns_order]

        df_normalized = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

        if classifier_train:
            X, index = create_sequences_predict(df_normalized, seq_length, classifier_train=True)
            return X, scaler, index

        else:
            X = create_sequences_predict(df_normalized, seq_length)
            return X, scaler

def sequence_generator(X_data, y_data, batch_size, shuffle=True):
    num_samples = len(X_data)
    indices = np.arange(num_samples)
    
    while True:
        if shuffle:
            np.random.shuffle(indices)
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            
            batch_indices = indices[start:end]
            
            yield X_data[batch_indices], y_data[batch_indices]
