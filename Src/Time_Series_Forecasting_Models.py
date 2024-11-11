import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pmdarima')
warnings.filterwarnings('ignore', category=FutureWarning)

# File paths
TSLA_PATH = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/portfolio-management-optimization/data/TSLA_processed.csv"
BND_PATH = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/portfolio-management-optimization/data/BND_processed.csv"
SPY_PATH = "C:/Users/hayyu.ragea/AppData/Local/Programs/Python/Python312/portfolio-management-optimization/data/SPY_processed.csv"

# Load data
tsla_data = pd.read_csv(TSLA_PATH, parse_dates=['Date'], index_col='Date')
bnd_data = pd.read_csv(BND_PATH, parse_dates=['Date'], index_col='Date')
spy_data = pd.read_csv(SPY_PATH, parse_dates=['Date'], index_col='Date')

# Ensure the data has a regular date frequency (daily)
tsla_data = tsla_data.asfreq('D', method='pad')  # Forward fill missing values if any

# Explore data structure
print(tsla_data.head())
print(tsla_data.describe())

# Plot TSLA closing price
plt.figure(figsize=(10, 6))
plt.plot(tsla_data['Close'], label='TSLA Close Price')
plt.title('Tesla Stock Price (TSLA)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Use the 'Close' price for forecasting
tsla_data = tsla_data[['Close']]

# Handle missing values if any
tsla_data = tsla_data.dropna()

# Train-test split
train_size = int(len(tsla_data) * 0.8)
train, test = tsla_data[:train_size], tsla_data[train_size:]

# Plot train and test sets
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.title('Train-Test Split for TSLA Stock Price')
plt.legend()
plt.show()

# Build and Train ARIMA Model
# Using Auto ARIMA to find best parameters (p, d, q)
model_arima = auto_arima(train, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)

# Fit ARIMA model
model_arima.fit(train)

# Forecasting
forecast_arima = model_arima.predict(n_periods=len(test))

# Ensure forecast_arima is the correct length
forecast_arima = pd.Series(forecast_arima, index=test.index)

# Handle NaN values before evaluation
test = test.dropna()
forecast_arima = forecast_arima.dropna()

# Check lengths match
print(len(test), len(forecast_arima))

# Evaluation metrics for ARIMA
if len(test) == len(forecast_arima):
    mae = mean_absolute_error(test, forecast_arima)
    rmse = np.sqrt(mean_squared_error(test, forecast_arima))
    mape = mean_absolute_percentage_error(test, forecast_arima)

    print(f"ARIMA Model Evaluation:\nMAE: {mae}\nRMSE: {rmse}\nMAPE: {mape}")
else:
    print("Error: The test set and forecast do not have matching lengths.")

# Plot ARIMA Forecast
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(forecast_arima, label='ARIMA Forecast', color='red')
plt.title('ARIMA Forecast for Tesla Stock Price')
plt.legend()
plt.show()

# Build and Train SARIMA Model
# Fit SARIMA model
model_sarima = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,5))  # Adjust parameters based on experimentation
model_sarima_fitted = model_sarima.fit(disp=False)

# Forecasting with SARIMA
forecast_sarima = model_sarima_fitted.predict(start=test.index[0], end=test.index[-1])

# Ensure the forecast_sarima index matches the test index
forecast_sarima.index = test.index

# Plot SARIMA Forecast
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(forecast_sarima, label='SARIMA Forecast', color='green')
plt.title('SARIMA Forecast for Tesla Stock Price')
plt.legend()
plt.show()

# Evaluation metrics for SARIMA
mae_sarima = mean_absolute_error(test, forecast_sarima)
rmse_sarima = np.sqrt(mean_squared_error(test, forecast_sarima))
mape_sarima = mean_absolute_percentage_error(test, forecast_sarima)

print(f"SARIMA Model Evaluation:\nMAE: {mae_sarima}\nRMSE: {rmse_sarima}\nMAPE: {mape_sarima}")

# Build and Train LSTM Model
# Prepare data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(tsla_data)

# Create dataset for LSTM (time step = 60)
X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

# Reshape X to be 3D for LSTM input
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_lstm.add(LSTM(units=50, return_sequences=False))
model_lstm.add(Dense(units=1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Train LSTM model
model_lstm.fit(X_train, y_train, epochs=10, batch_size=32)

# Make predictions with LSTM
predictions_lstm = model_lstm.predict(X_test)

# Inverse transform the predictions
predictions_lstm = scaler.inverse_transform(predictions_lstm)

# Evaluate LSTM model
mae_lstm = mean_absolute_error(scaler.inverse_transform(y_test.reshape(-1, 1)), predictions_lstm)
rmse_lstm = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), predictions_lstm))
mape_lstm = mean_absolute_percentage_error(scaler.inverse_transform(y_test.reshape(-1, 1)), predictions_lstm)

# Corrected LSTM evaluation print statement
print(f"LSTM Model Evaluation:\nMAE: {mae_lstm}\nRMSE: {rmse_lstm}\nMAPE: {mape_lstm}")

# Plot LSTM Forecast
plt.figure(figsize=(10, 6))

# Adjust the test index to match y_test length
test_index = tsla_data.index[train_size:train_size + len(y_test)]  # Adjust to match y_test length

# Plot test data and LSTM forecast
plt.plot(test_index, scaler.inverse_transform(y_test.reshape(-1, 1)), label='Test')
plt.plot(test_index, predictions_lstm, label='LSTM Forecast', color='orange')

plt.title('LSTM Forecast for Tesla Stock Price')
plt.legend()
plt.show()
