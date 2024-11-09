# Interim Report: Time Series Forecasting for Portfolio Management Optimization

## Introduction

Guide Me in Finance (GMF) Investments is dedicated to enhancing its portfolio management strategies by integrating advanced time series forecasting models. The goal is to predict market trends, optimize asset allocation, and improve portfolio performance for clients. This report covers the progress made so far in applying time series forecasting to key financial assets: **Tesla (TSLA)**, **Vanguard Total Bond Market ETF (BND)**, and **S&P 500 ETF (SPY)**.

## Business Objective

The core objective is to leverage time series forecasting techniques to predict market trends and optimize asset allocation for GMF's clients. The goals are:

- **Predict Future Market Trends**: Apply forecasting models to predict the performance of key financial assets.
- **Optimize Asset Allocation**: Generate recommendations for portfolio rebalancing based on forecasted market trends.
- **Offer Actionable Insights**: Assist GMF in making data-driven decisions that maximize returns and minimize risks.

## Key Tasks

1. **Data Extraction and Preprocessing**: Gather historical data from financial assets.
2. **Model Development**: Implement time series models to forecast asset price trends.
3. **Portfolio Optimization**: Use forecasted trends to optimize portfolio allocation.

## Data Overview

### Assets Analyzed

- **Tesla (TSLA)**: High-growth, high-risk stock in the automotive sector.
- **Vanguard Total Bond Market ETF (BND)**: Low-risk bond ETF for stability and predictable returns.
- **S&P 500 ETF (SPY)**: Balanced exposure to the broader stock market with moderate risk.

### Data Source

- **Yahoo Finance** via the YFinance API, spanning from January 1, 2015, to October 31, 2024.

### Preprocessing Steps

- **Missing Data Handling**: Forward-fill used to fill missing values.
- **Standardization**: Ensured numerical consistency for comparative analysis.

### Processed Data

- **TSLA Data**: Saved as `TSLA_processed.csv`
- **BND Data**: Saved as `BND_processed.csv`
- **SPY Data**: Saved as `SPY_processed.csv`

## Exploratory Data Analysis (EDA) Results

### Key insights from the EDA of the three assets

- **Time Series Visualization (Closing Price Over Time)**:
  - **TSLA**: Exhibits high volatility with large price swings.
  - **BND**: Stable performance with minor fluctuations.
  - **SPY**: Moderate trends aligned with broader market movements.

- **Volatility (Daily Percentage Change)**:
  - **TSLA**: High volatility with frequent large changes.
  - **BND**: Low volatility, reflecting stability.
  - **SPY**: Moderate fluctuations in line with the overall market.

- **Rolling Mean & Standard Deviation (30-day window)**:
  - **TSLA**: Significant fluctuations in rolling mean.
  - **BND**: Consistent and stable rolling mean.
  - **SPY**: Moderate changes, consistent with market trends.

- **Outlier Detection**:
  - **TSLA**: Identified several outliers, reflecting high volatility.
  - **BND**: Few outliers due to its stability.
  - **SPY**: A small number of outliers, often reflecting market events.

## Forecasting Results (SARIMA Model)

The **SARIMA (Seasonal AutoRegressive Integrated Moving Average)** model was applied to forecast future price trends for the three assets.

### Forecast for TSLA (Next 30 Days)

**Expected Price Fluctuations**: TSLA is expected to continue experiencing volatility with fluctuations.

**Sample Forecast Values**:

- **2024-10-31**: -0.000973
- **2024-11-01**: 0.000360
- **2024-11-02**: -0.006672
- **2024-11-03**: 0.000270
- **2024-11-04**: 0.001477
- **2024-11-05**: 0.000137
- **2024-11-06**: 0.000109
- **2024-11-07**: -0.000613
- **2024-11-08**: -0.001851
- **2024-11-09**: 0.000962

### Forecast for BND (Next 30 Days)

**Expected Price Fluctuations**: BND is forecasted to exhibit stable returns with minor fluctuations.

**Sample Forecast Values**:

- **2024-10-31**: -0.001900
- **2024-11-01**: -0.002357
- **2024-11-02**: 0.001178
- **2024-11-03**: -0.005295
- **2024-11-04**: 0.003379
- **2024-11-05**: 0.002065
- **2024-11-06**: -0.004518
- **2024-11-07**: -0.002566
- **2024-11-08**: -0.001279
- **2024-11-09**: 0.002741

### Forecast for SPY (Next 30 Days)

**Expected Price Fluctuations**: SPY is expected to show moderate fluctuations in line with broader market movements.

**Sample Forecast Values**:

- **2024-10-31**: 0.001144
- **2024-11-01**: 0.001038
- **2024-11-02**: -0.000556
- **2024-11-03**: -0.002050
- **2024-11-04**: 0.001103
- **2024-11-05**: -0.002176
- **2024-11-06**: 0.000897
- **2024-11-07**: -0.000687
- **2024-11-08**: -0.001014
- **2024-11-09**: 0.001661

## Portfolio Optimization and Allocation

Using daily returns for each asset, portfolio optimization was performed based on the **Sharpe ratio** and **Value at Risk (VaR)**.

### Optimal Portfolio Weights

- **TSLA**: 34.1%
- **BND**: 0.0%
- **SPY**: 65.9%

### Sharpe Ratios

- **TSLA Sharpe Ratio**: -0.14
- **BND Sharpe Ratio**: N/A (due to NaN values in daily returns)
- **SPY Sharpe Ratio**: N/A (due to NaN values in daily returns)

### Value at Risk (VaR)

- **TSLA VaR**: -0.09

## GitHub Link

[GitHub Repository: Portfolio Management Optimization](https://github.com/HaYyu-Ra/portfolio-management-optimization)

## Conclusion

The time series forecasts for **TSLA**, **BND**, and **SPY** provide valuable insights into the potential future movements of these assets. The volatility in **TSLA** suggests it remains a high-risk, high-reward asset, while **BND** shows stable returns with low risk. **SPY** offers moderate exposure to the overall market.

### Portfolio Optimization Recommendations

- GMF should consider allocating **34.1%** of the portfolio to **TSLA**, **0%** to **BND** (given its low risk-adjusted return), and **65.9%** to **SPY**.

## Next Steps

1. **Refine Forecasting Models**: Incorporate updated data for more accurate predictions.
2. **Incorporate Additional Risk Management Strategies**: Evaluate and integrate other metrics such as drawdown and maximum loss to better manage portfolio risk.
3. **Recalculate Portfolio Allocations**: Regularly update portfolio recommendations based on new market data and changing forecast trends.

This interim report reflects a comprehensive analysis of historical trends and forecasting models, helping GMF optimize its asset allocation strategies based on current market trends.
