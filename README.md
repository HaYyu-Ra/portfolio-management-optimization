# Portfolio Management Optimization

This project aims to build an optimized portfolio management system using various financial analysis techniques, including data fetching, preprocessing, exploratory data analysis (EDA), forecasting, and portfolio optimization. The goal is to assist in making investment decisions by optimizing asset allocation using modern financial algorithms.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started with this project, follow these steps:

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/HaYyu-Ra/portfolio-management-optimization.git
    ```

2. Navigate to the project directory:

    ```bash
    cd portfolio-management-optimization
    ```

3. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4. Install the necessary dependencies from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

To start using the project:

1. **Data Fetching**: The `scripts/data_fetch.py` file allows you to fetch the latest financial data for portfolio optimization.
2. **EDA**: Use the Jupyter notebooks in the `notebooks/` directory to perform exploratory data analysis (EDA) on the financial data.
3. **Forecasting**: Run the forecasting models from the `scripts/forecasting.py` to predict the future trends of the assets.
4. **Optimization**: The portfolio optimization module in `scripts/portfolio.py` optimizes the asset allocation based on the forecasted data.

Run the scripts or notebooks as needed to perform the desired analysis or optimizations.

## Directory Structure

The project follows the structure below:
portfolio-management-optimization/ ├── notebooks/ # Jupyter notebooks for EDA and forecasting ├── scripts/ # Python scripts for data fetching, preprocessing, and optimization ├── tests/ # Unit tests ├── .github/ # GitHub Actions workflows ├── .gitignore # Git ignore rules ├── requirements.txt # Project dependencies ├── README.md # Project documentation

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository: [Portfolio Management Optimization](https://github.com/HaYyu-Ra/portfolio-management-optimization).
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
