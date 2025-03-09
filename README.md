# AI-supported Forecasting Models for Waste Management and Resource Planning

This project develops a forecasting system to optimize waste processing in waste incineration plants. The system predicts arrival times, quantities, and quality scores of waste, considering seasonal fluctuations and other characteristics. It aims to enable more efficient resource planning in disposal facilities and improve processing procedures.

## Project Structure

### Notebooks
- **Data Examination Notebooks**: Explore trends, seasonality, and statistics in waste data to inform feature engineering.
  - `quantity.ipynb`: Descriptive analysis of the quantity of delivered waste.
  - `quality.ipynb`: Analysis of the quality scores per delivery.
- **Forecasting Notebooks**: Develop and test forecasting models.
  - `arrival_time.ipynb`: Forecast the arrival times of waste deliveries.
  - `quantity_xgboost.ipynb`: Forecast waste quantities using the XGBoost algorithm.
  - `quality_regression_test.ipynb`: Regression tests for quality score predictions.
  - `forecasting_system.ipynb`: Integrates various forecasting models for comprehensive analysis.
- **Implementation Tests**: Test new features and forecasting methods.
  - `test_prediction_intervals.ipynb`
  - `test_prophet.ipynb`

### Modules
- **Data Preparation**: Contains tools for preparing data for forecasting tasks.
  - `data_processor.py`: Includes functions for data cleaning, preprocessing, and feature extraction.

### Models
- **Hyperparameter Search Results**: Contains results and configurations for various models.
  - `hyperparameters.py`: Details of hyperparameter tuning for the XGBoost model.

## Prerequisites

- **Python Version**: This project was build using Python 3.12.4. Please ensure you have this version installed before setting up the project environment.

## Installation

Clone this repository and install the required packages:

```bash
git clone https://github.com/1106maxi/AI-supported-Forecasting-Models-for-Waste-Management.git
cd AI-supported-Forecasting-Models-for-Waste-Management
pip install -r requirements.txt
