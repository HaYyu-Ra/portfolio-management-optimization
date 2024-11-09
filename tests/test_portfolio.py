import pytest
import pandas as pd
from scripts.portfolio import portfolio_allocation

# Helper function to create sample data
def create_sample_data():
    # Create sample data for TSLA, BND, and SPY
    dates = pd.date_range(start='2023-01-01', periods=5, freq='B')
    
    tsla_data = pd.DataFrame({
        'Date': dates,
        'Close': [700, 710, 715, 720, 725]
    }).set_index('Date')
    tsla_data['Daily Change'] = tsla_data['Close'].pct_change()  # Calculate daily change
    
    bnd_data = pd.DataFrame({
        'Date': dates,
        'Close': [100, 101, 102, 103, 104]
    }).set_index('Date')
    bnd_data['Daily Change'] = bnd_data['Close'].pct_change()  # Calculate daily change
    
    spy_data = pd.DataFrame({
        'Date': dates,
        'Close': [400, 405, 410, 415, 420]
    }).set_index('Date')
    spy_data['Daily Change'] = spy_data['Close'].pct_change()  # Calculate daily change
    
    return tsla_data, bnd_data, spy_data

# Test case for portfolio allocation
def test_portfolio_allocation():
    tsla_data, bnd_data, spy_data = create_sample_data()

    # Calculate portfolio allocation
    weights = portfolio_allocation(tsla_data, bnd_data, spy_data)

    # Check if the allocation is returned as expected
    assert 'TSLA' in weights
    assert 'BND' in weights
    assert 'SPY' in weights

    # Check if the total allocation adds up to 1 (this is a simple check)
    assert round(weights['TSLA'] + weights['BND'] + weights['SPY'], 2) == 1.0

