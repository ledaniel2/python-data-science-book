# Chapter 14: Time Series Analysis with Python

Time series analysis ia a crucial aspect of data science that deals with data points collected or indexed over time. Time series data is ubiquitous, spanning various domains such as finance, economics, meteorology, and healthcare. Analyzing time series data allows us to uncover trends, seasonality, and other patterns, helping us make informed decisions and predictions.

This chapter will begin by discussing how to handle time-stamped data using Python, covering essential concepts such as date and time objects, time zones, and datetime arithmetic. We will then explore time series decomposition, a technique that breaks down a time series into its constituent components, such as trend, seasonality, and residual noise. Next, we will introduce various forecasting techniques, ranging from classical methods like moving averages and exponential smoothing to more advanced approaches like ARIMA and SARIMAX.

Throughout this chapter, you will work with real-world time series datasets and leverage Python libraries such as `pandas` and `statsmodels` to perform time series analysis and forecasting tasks.

Our learning goals for this chapter are:

 * Learn the fundamentals of time series data, its characteristics, and its importance in various fields.
 * Master techniques to work with date and time objects, time zones, and datetime arithmetic using Python.
 * Learn how to break down a time series into its constituent components, such as trend, seasonality, and residual noise.
 * Become familiar with various forecasting methods, from classical techniques like moving averages and exponential smoothing to advanced approaches like ARIMA and SARIMAX.
 * Gain hands-on experience in time series analysis and forecasting using popular Python libraries such as `pandas` and `statsmodels`.

## 14.1: Handling Time-stamped Data

Time-stamped data refers to data that is collected at regular or irregular intervals over a period of time. This type of data is quite common in various fields, such as finance, weather forecasting, and web analytics. By analyzing time series data, we can uncover patterns, trends, and make predictions about future events.

To work with time series data in Python, we'll be using the `pandas` library, which provides powerful tools for handling and manipulating time-stamped data.

First, we need to import the necessary libraries:

```python
import pandas as pd
import numpy as np
```

Now let's create a simple time series dataset to work with. For this example, we will use a dataset containing daily temperatures for a year:

```python
# Generate a range of dates
date_rng = pd.date_range(start='1/1/2022', end='12/31/2022', freq='D')

# Create a time series DataFrame with random temperature values
temperature_data = np.random.randint(50, 100, size=(len(date_rng)))
temp_df = pd.DataFrame(date_rng, columns=['date'])
temp_df['temperature'] = temperature_data

print(temp_df.head())
```

This will output:

```plaintext
        date  temperature
0 2022-01-01           64
1 2022-01-02           89
2 2022-01-03           74
3 2022-01-04           54
4 2022-01-05           67
```

Now that we have a time series dataset, let's explore some common operations for handling time-stamped data.

### Setting the DatetimeIndex

One of the first things we should do is set the `date` column as the index of our DataFrame. This will make it easier to manipulate and analyze our time series data:

```python
temp_df.set_index('date', inplace=True)
print(temp_df.head())
```

This will output:

```plaintext
            temperature
date                   
2022-01-01           64
2022-01-02           89
2022-01-03           74
2022-01-04           54
2022-01-05           67
```

### Resampling Time Series Data

Resampling is the process of aggregating time series data over different time intervals. With `pandas`, we can easily resample our data using the `resample()` function. For example, let's calculate the monthly average temperature for our dataset:

```python
monthly_avg_temp = temp_df.resample('M').mean()
print(monthly_avg_temp)
```

This will output:

```plaintext
            temperature
date                   
2022-01-31    74.903226
2022-02-28    74.571429
...
2022-12-31    74.096774
```

### Slicing Time Series Data

To extract specific time intervals from our dataset, we can use the slice notation with the `DatetimeIndex`. For example, let's extract the temperature data for February:

```python
feb_temp = temp_df['2022-02-01':'2022-02-28']
print(feb_temp)
```

This will output:

```plaintext
            temperature
date                   
2022-02-01           91
2022-02-02           81
...
2022-02-28           60
```

### Shifting Time Series Data

Shifting is another useful operation in time series analysis, which allows us to move the data points forward or backward in time. This can be helpful in calculating differences or changes over time. In `pandas`, we can use the `shift()` function to achieve this. Let's calculate the day-to-day change in temperature:

```python
temp_df['previous_day_temp'] = temp_df['temperature'].shift(1)
temp_df['daily_change'] = temp_df['temperature'] - temp_df['previous_day_temp']
print(temp_df.head())
```

This will output:

```plaintext
            temperature  previous_day_temp  daily_change
date                                                    
2022-01-01           64                NaN           NaN
2022-01-02           89               64.0          25.0
2022-01-03           74               89.0         -15.0
2022-01-04           54               74.0         -20.0
2022-01-05           67               54.0          13.0
```

### Handling Missing Data

Time series data can sometimes contain missing values due to various reasons, such as data collection errors or gaps in the data. To handle missing data in `pandas`, we can use the `fillna()` function along with various methods like forward fill (`ffill`) or backward fill (`bfill`). For example, let's create a small gap in our temperature data and fill it using forward fill:

```python
# Introduce a gap in the data
temp_df.iloc[7:10, 0] = np.nan

# Fill the gap using forward fill
temp_df['temperature_filled'] = temp_df['temperature'].fillna(method='ffill')
print(temp_df.head(12))
```

This will output:

```plaintext
            temperature  previous_day_temp  daily_change  temperature_filled
date                                                                         
2022-01-01           64                NaN           NaN                  64
2022-01-02           89               64.0          25.0                  89
...
2022-01-08          NaN               78.0           NaN                  78
2022-01-09          NaN                NaN           NaN                  78
2022-01-10          NaN                NaN           NaN                  78
2022-01-11           84                NaN           NaN                  84
```

In summary, we have covered the basics of handling time-stamped data using the `pandas` library in Python. By understanding these fundamental operations, you will be well-prepared to evaluate time series analysis and forecasting techniques in the rest of this chapter.

## 14.2: Time Series Decomposition

Decomposition is a technique used to break down a time series into its individual components, which can help us better understand the underlying patterns and make more accurate forecasts.

A time series can typically be decomposed into three main components:

 1. Trend: The long-term movement or underlying direction of the series.
 2. Seasonality: The recurring patterns or fluctuations that occur periodically, such as daily, weekly, or annually.
 3. Residual: The random variation or noise that cannot be explained by the trend and seasonality.

Let's go through each component and explore how to perform time series decomposition using Python.

### Importing Libraries

First, we need to import the necessary libraries. We will be using `pandas` for handling our data and `matplotlib` for plotting. We will also use the `statsmodels` library, which provides a method called `seasonal_decompose()` for time series decomposition.

The `statsmodels` library can be installed with the following command in a terminal or command window:

```bash
pip install statsmodels
```

Here are the libraries we will use:

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
```

### Loading the Data

For this example, we will use a dataset containing monthly airline passenger data. This dataset can be downloaded from the following link: https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv

We can load the dataset using `pandas`:

```python
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
data = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
```

In the code above, we set the `parse_dates` parameter to recognize the "Month" column as a date and the `index_col` parameter to set the "Month" column as the index of the DataFrame.

### Visualizing the Data

Before decomposing the time series, it's helpful to visualize the data. We can do this using `matplotlib`:

```python
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.xlabel('Month')
plt.ylabel('Number of Passengers')
plt.title('Airline Passengers (1949-1960)')
plt.show()
```

From the plot, we can see that there is an upward trend in the number of airline passengers, as well as some seasonality.

### Decomposing the Time Series

Now we are ready to decompose the time series. We will use the `seasonal_decompose()` function from the `statsmodels` library:

```python
decomposition = seasonal_decompose(data, model='multiplicative')
```

In the code above, we pass the data DataFrame and set the model parameter to "multiplicative". The other option is "additive", which would be more appropriate for time series data with constant seasonality.
The `seasonal_decompose()` function returns an object with attributes for the trend, seasonal, and residual components. We can visualize these components using `matplotlib`:

```python
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
ax1.plot(data)
ax1.set_title('Original Time Series')
ax1.set_xlabel('Month')
ax1.set_ylabel('Number of Passengers')

ax2.plot(decomposition.trend)
ax2.set_title('Trend')
ax2.set_xlabel('Month')
ax2.set_ylabel('Number of Passengers')

ax3.plot(decomposition.seasonal)
ax3.set_title('Seasonal')
ax3.set_xlabel('Month')
ax3.set_ylabel('Number of Passengers')

ax4.plot(decomposition.resid)
ax4.set_title('Residual')
ax4.set_xlabel('Month')
ax4.set_ylabel('Number of Passengers')

plt.tight_layout()
plt.show()
```

In the resulting plots, we can observe the individual components of the time series:

 1. Original Time Series: This is the original dataset showing the number of airline passengers over time.
 2. Trend: The upward trend in the number of passengers is clearly visible. This component represents the long-term increase in the data.
 3. Seasonal: The seasonal component shows a repeating pattern over the course of each year. This represents the regular fluctuations in the number of passengers due to factors such as holidays or seasonal travel.
 4. Residual: The residual component represents the random variation or noise in the data that cannot be explained by the trend and seasonal components. We can use this information to detect anomalies or further refine our forecasting models.

In summary, we have learned about time series decomposition and how to break down a time series into its main components: trend, seasonality, and residual. We explored how to perform decomposition using Python, and visualized the individual components to better understand the underlying patterns in the data. This knowledge will be useful when building forecasting models later in this chapter.

## 14.3: Forecasting Techniques

We will explore various forecasting techniques for time series data analysis. Forecasting is the process of predicting future data points based on historical data. Time series forecasting is used in various domains, such as finance, economics, and meteorology, to make informed decisions and predictions. We will cover the following techniques:

 1. Naive and Seasonal Naive Methods
 2. Moving Averages
 3. Exponential Smoothing
 4. Autoregression (AR)
 5. Moving Average (MA)
 6. Autoregressive Integrated Moving Average (ARIMA)
 7. Seasonal Decomposition of Time Series (STL) and Seasonal ARIMA (SARIMA)

First, let's import the necessary libraries and load a sample time series dataset from https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
data = pd.read_csv(url, index_col=0, parse_dates=True)

# Plot the time series data
data.plot(figsize=(12, 6))
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Daily Minimum Temperatures in Melbourne')
plt.show()
```

 1. Naive and Seasonal Naive Methods: The naive method simply assumes that the future value is equal to the most recent observed value. In the seasonal naive method, we assume that the future value is equal to the value from the last season.

```python
def naive_forecast(data, periods=1):
    return data.shift(periods)

def seasonal_naive_forecast(data, periods=1, season_length=1):
    return data.shift(season_length * periods)
```

 2. Moving Averages: Moving averages calculate the average of a rolling window of data points. Simple moving averages (SMA) and weighted moving averages (WMA) are common techniques.

```python
def simple_moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

def weighted_moving_average(data, weights):
    return data.rolling(window=len(weights)).apply(lambda x: np.sum(weights * x) / np.sum(weights), raw=True)
```

 3. Exponential Smoothing: Exponential smoothing techniques assign exponentially decreasing weights to past observations. Single, double, and triple exponential smoothing are popular methods.

```python
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
```

 4. Autoregression (AR): The autoregression model assumes that the value at the current time step depends linearly on the values at previous time steps.

```python
from statsmodels.tsa.ar_model import AutoReg

def autoregression(data, lag=1):
    model = AutoReg(data, lags=lag)
    model_fit = model.fit()
    return model_fit.fittedvalues
```

 5. Moving Average (MA): The moving average model assumes that the value at the current time step depends linearly on the past errors.

```python
from statsmodels.tsa.arima.model import ARIMA

def moving_average(data, order=(0, 0, 1)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit.fittedvalues
```

 6. Autoregressive Integrated Moving Average (ARIMA): ARIMA is a combination of the autoregression (AR) and moving average (MA) models. The 'integrated' part refers to the differencing of raw observations to stabilize the time series data.

```python
def arima(data, order=(1, 1, 1)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit.fittedvalues
```

 7. Seasonal Decomposition of Time Series (STL) and Seasonal ARIMA (SARIMA): STL decomposes a time series into three components: trend, seasonality, and residuals. SARIMA is an extension of the ARIMA model that considers seasonality in the time series data.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

def seasonal_arima(data, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)):
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    return model_fit.fittedvalues
```

Here's how to apply these techniques to our sample dataset:

```python
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
```

To evaluate the performance of these forecasting techniques, you can use metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), or Root Mean Squared Error (RMSE). These metrics help you to compare different models and choose the one that works best for your dataset.

```python
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
```

This will output:

```plaintext
Naive: MAE = 2.14, MSE = 7.57, RMSE = 2.75
SARIMA: MAE = 1.94, MSE = 6.18, RMSE = 2.49
```

Remember that time series data is often subject to different conditions and influences over time. Therefore, it's essential to continually evaluate and adjust your models to maintain their accuracy and reliability. As you gain more experience with time series forecasting, you'll become more adept at selecting the right techniques for your data and problem.
