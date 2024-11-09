import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Fill missing values with forward fill, then drop any remaining missing values
    data = data.fillna(method="ffill").dropna()
    # Scale the specified columns
    scaler = StandardScaler()
    data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = scaler.fit_transform(
        data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    )
    return data

if __name__ == "__main__":
    # Define the full paths for each ticker's CSV data file
    data_paths = {
        "TSLA": r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\portfolio-management-optimization\data\TSLA_data.csv",
        "BND": r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\portfolio-management-optimization\data\BND_data.csv",
        "SPY": r"C:\Users\hayyu.ragea\AppData\Local\Programs\Python\Python312\portfolio-management-optimization\data\SPY_data.csv"
    }

    for ticker, path in data_paths.items():
        # Load the data
        data = pd.read_csv(path, index_col="Date", parse_dates=True)
        # Process the data
        processed_data = preprocess_data(data)
        # Save the processed data to the same directory
        processed_path = path.replace("_data.csv", "_processed.csv")
        processed_data.to_csv(processed_path)
