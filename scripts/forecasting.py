import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import timedelta

# Load your data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    return data

# Apply differencing to make data stationary if needed
def make_stationary(data):
    return data.diff().dropna()

# Decompose the time series into trend, seasonal, and residual components (Additive Model)
def seasonal_decompose_data_additive(data):
    # Use the additive model since the data contains negative or zero values
    decomposition = seasonal_decompose(data, model='additive', period=252)  # Assuming 252 trading days
    trend = decomposition.trend.dropna()
    seasonal = decomposition.seasonal.dropna()
    residual = decomposition.resid.dropna()
    
    # Plot decomposition components
    plt.figure(figsize=(12, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(data, label='Original Data')
    plt.title('Original Time Series')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(trend, label='Trend', color='orange')
    plt.title('Trend Component')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(seasonal, label='Seasonal', color='green')
    plt.title('Seasonal Component')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(residual, label='Residual', color='red')
    plt.title('Residual Component')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    return trend, seasonal, residual

# Decompose the time series into trend, seasonal, and residual components (Multiplicative Model)
def seasonal_decompose_data_multiplicative(data):
    # Shift the data to ensure all values are positive
    if data.min() <= 0:
        data = data + abs(data.min()) + 1  # Add the absolute value of the minimum value + 1
    
    decomposition = seasonal_decompose(data, model='multiplicative', period=252)  # Assuming 252 trading days
    trend = decomposition.trend.dropna()
    seasonal = decomposition.seasonal.dropna()
    residual = decomposition.resid.dropna()
    
    # Plot decomposition components
    plt.figure(figsize=(12, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(data, label='Original Data')
    plt.title('Original Time Series')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(trend, label='Trend', color='orange')
    plt.title('Trend Component')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(seasonal, label='Seasonal', color='green')
    plt.title('Seasonal Component')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(residual, label='Residual', color='red')
    plt.title('Residual Component')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    return trend, seasonal, residual

# Train SARIMA model
def train_sarima_model(data, p, d, q, P, D, Q, s):
    # Fit SARIMA model: (p, d, q) x (P, D, Q, s)
    model = SARIMAX(data, order=(p, d, q), seasonal_order=(P, D, Q, s))
    results = model.fit(maxiter=1000, method='bfgs')  # Increase iterations and use BFGS
    return results

# Forecast future values
def forecast_sarima(model, steps):
    forecast = model.get_forecast(steps=steps)
    forecast_mean = forecast.predicted_mean
    return forecast_mean

# Visualize the results
def visualize_forecast(data, forecast, ticker):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data, label='Actual Data', color='blue')
    plt.plot(forecast.index, forecast, label='Forecast', color='red')
    plt.title(f'{ticker} Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Main forecasting function
def main():
    tickers = ['TSLA', 'BND', 'SPY']
    file_paths = {
        'TSLA': r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\portfolio-management-optimization\data\TSLA_processed.csv",
        'BND': r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\portfolio-management-optimization\data\BND_processed.csv",
        'SPY': r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\portfolio-management-optimization\data\SPY_processed.csv"
    }

    for ticker in tickers:
        print(f"Forecast for {ticker}:")
        data = load_data(file_paths[ticker])

        # Decompose the time series into trend, seasonal, and residual components
        trend, seasonal, residual = seasonal_decompose_data_additive(data['Close'])  # Use Additive Model

        # Optionally, you can apply SARIMA on the residuals (which remove seasonality and trend)
        data_stationary = make_stationary(residual)  # Using residuals for SARIMA

        # Train SARIMA model on residuals
        results = train_sarima_model(data_stationary, p=1, d=1, q=1, P=1, D=1, Q=1, s=5)  # Example SARIMA order

        # Forecast future values (e.g., 30 days ahead)
        forecast = forecast_sarima(results, steps=30)

        # Visualize the forecast
        forecast.index = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=30)
        visualize_forecast(data['Close'], forecast, ticker)

        # Print the forecast
        print(forecast)
        print("\n")

if __name__ == "__main__":
    main()
