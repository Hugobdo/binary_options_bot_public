import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from mongo_connector import MongoData

mongo = MongoData()
mongo.update_log(verbose=True)

# all logs with a close price
log_df = pd.DataFrame(list(mongo.log_collection.find({"close": {"$exists": True}})))
#round predicted price to 6 decimal places
log_df['predicted_price'] = log_df['predicted_price'].round(6)
log_df['close'] = log_df['close'].round(6)

def evaluate_row(row):

    if row['current_price'] < row['predicted_price']:  # 'call'
        if row['current_price'] < row['close']:
            return 'win'
        elif row['current_price'] == row['close']:
            return 'equal'
        else:
            return 'loss'

    elif row['current_price'] > row['predicted_price']:  # 'put'
        if row['current_price'] > row['close']:
            return 'win'
        elif row['current_price'] == row['close']:
            return 'equal'
        else:
            return 'loss'

    else:  # Caso o current_price seja igual ao predicted_price
        return 'neutral'  # ou qualquer tratamento que você queira dar nesse caso

def distance_correction(row):

    if row['current_price'] < row['predicted_price']:  # 'call'
        return row['predicted_lower_bound'] - row['current_price']
    
    elif row['current_price'] > row['predicted_price']:  # 'put'
        return row['predicted_upper_bound'] - row['current_price']
    
    else:  # Caso o current_price seja igual ao predicted_price
        return 0  # ou qualquer tratamento que você queira dar nesse caso

def prediction_spread(row):
    return (row['predicted_upper_bound'] - row['predicted_lower_bound'])/2

log_df['result'] = log_df.apply(evaluate_row, axis=1)
log_df['bounds_residual'] = log_df.apply(distance_correction, axis=1)
log_df['prediction_spread'] = log_df.apply(prediction_spread, axis=1)
#mean spread where action is not Pass
log_df['action'].value_counts()
#result value counts where action is not Pass
log_df[log_df['action']!='Pass']['result'].value_counts(normalize=True)

# accuracy
log_df['result'].value_counts(normalize=True)

# Residuals
log_df['residual'] = log_df['close'] - log_df['predicted_price']

# # Plot residuals
# log_df['residual'].plot()

#residual MAE e RMSE
mean_absolute_error(log_df['residual'], np.zeros(len(log_df)))  
np.sqrt(mean_squared_error(log_df['residual'], np.zeros(len(log_df))))

# residual mae for each result, last 1000 rows
log_df.groupby('result')['residual'].apply(lambda x: mean_absolute_error(x, np.zeros(len(x))))

# residual distribution
log_df['residual'].hist(bins=100)

# residual ACF on seq_length=20
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(log_df[log_df['seq_length']==20]['residual'], lags=100)

#PACF
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(log_df['residual'], lags=100)

# plot dispersion of confidence level and lower bound
log_df.plot.scatter(x='confidence_level', y='predicted_upper_bound')

#value counts where action not Pass
