import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
data = pd.read_csv(url, index_col=0, parse_dates=True)

# Plot the time series data
#data.plot(figsize=(12, 6))
#plt.xlabel('Date')
#plt.ylabel('Temperature')
#plt.title('Daily Minimum Temperatures in Melbourne')
#plt.show()

def naive_forecast(data, periods=1):
    return data.shift(periods)

def seasonal_naive_forecast(data, periods=1, season_length=1):
    return data.shift(season_length * periods)

def simple_moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

def weighted_moving_average(data, weights):
    return data.rolling(window=len(weights)).apply(lambda x: np.sum(weights * x) / np.sum(weights), raw=True)

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

# Single Exponential Smoothing
def single_exp_smoothing(data, smoothing_level=None):
    model = SimpleExpSmoothing(data)
    model_fit = model.fit(smoothing_level=smoothing_level)
    return model_fit.fittedvalues

# Double and Triple Exponential Smoothing
def double_and_triple_exp_smoothing(data, seasonal_periods=None, trend=None, seasonal=None):
    model = ExponentialSmoothing(data, seasonal_periods=seasonal_periods, trend=trend, seasonal=seasonal)
    model_fit = model.fit()
    return model_fit.fittedvalues

from statsmodels.tsa.ar_model import AutoReg

def autoregression(data, lag=1):
    model = AutoReg(data, lags=lag)
    model_fit = model.fit()
    return model_fit.fittedvalues

from statsmodels.tsa.arima.model import ARIMA

def moving_average(data, order=(0, 0, 1)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit.fittedvalues

def arima(data, order=(1, 1, 1)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit.fittedvalues

from statsmodels.tsa.statespace.sarimax import SARIMAX

def seasonal_arima(data, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)):
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    return model_fit.fittedvalues

# Calculate forecasts using various methods
naive_forecast_data = naive_forecast(data)
seasonal_naive_forecast_data = seasonal_naive_forecast(data, season_length=365)
sma_data = simple_moving_average(data, window_size=7)
wma_data = weighted_moving_average(data, weights=np.array([0.1, 0.2, 0.3, 0.4]))
single_exp_data = single_exp_smoothing(data)
double_exp_data = double_and_triple_exp_smoothing(data, trend='add')
triple_exp_data = double_and_triple_exp_smoothing(data, seasonal_periods=365, trend='add', seasonal='add')
ar_data = autoregression(data)
ma_data = moving_average(data)
arima_data = arima(data)
sarima_data = seasonal_arima(data)

# Plot the original time series and forecasts
plt.figure(figsize=(12, 6))
plt.plot(data, label='Original')
plt.plot(naive_forecast_data, label='Naive')
plt.plot(seasonal_naive_forecast_data, label='Seasonal Naive')
plt.plot(sma_data, label='SMA')
plt.plot(wma_data, label='WMA')
plt.plot(single_exp_data, label='Single Exp')
plt.plot(double_exp_data, label='Double Exp')
plt.plot(triple_exp_data, label='Triple Exp')
plt.plot(ar_data, label='AR')
plt.plot(ma_data, label='MA')
plt.plot(arima_data, label='ARIMA')
plt.plot(sarima_data, label='SARIMA')
plt.legend(loc='upper left')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Daily Minimum Temperatures in Melbourne - Forecasting Techniques')
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

# Fill missing values from the forecasted data
data_clean = data.fillna(0.0)
naive_forecast_clean = naive_forecast_data.fillna(0.0)
sarima_forecast_clean = sarima_data.fillna(0.0)

# Evaluate the performance of the naive and seasonal naive methods
mae_naive, mse_naive, rmse_naive = evaluate_forecast(data_clean, naive_forecast_clean)
mae_sarima, mse_sarima, rmse_sarima = evaluate_forecast(data_clean, sarima_forecast_clean)

print(f'Naive: MAE = {mae_naive:.2f}, MSE = {mse_naive:.2f}, RMSE = {rmse_naive:.2f}')
print(f'SARIMA: MAE = {mae_sarima:.2f}, MSE = {mse_sarima:.2f}, RMSE = {rmse_sarima:.2f}')
