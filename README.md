# AI-supported Forecasting Models for Waste Management and Resource Planning

Modern waste incineration plants play a vital role in sustainable waste management by reducing volume and recovering energy. However, due to the heterogeneous nature of waste, fluctuations in fuel quality can lead to operational inefficiencies, increased auxiliary fuel use, and higher emissions.

To ensure stable combustion, reduce costs, and meet regulatory standards (e.g., EU Directive 2010/75/EC), accurate forecasting of key input variables, such as waste quantity, quality, and delivery timing, is essential.

This repository contains the code for my seminar project, which explores AI-based forecasting methods for waste incineration plant operations using a synthetic dataset derived from real-world process data.

**Main goals:**
- Forecast daily waste quantities using Facebook Prophet and XGBoost
- Identify effective feature sets and model structures (e.g., AR, NAR, ARR)
- Compare models with baseline approaches like Holt-Winters and validate results

While the seminar paper focused on **waste quantity forecasting**, this repository includes supporting analysis and experiments for all three targets: **quantity, quality, and arrival time**.

## Project Structure

```
├── data_exploration/                         
│   ├── arrival_time_analysis.ipynb           # EDA for arrival time data  (not included in seminar paper)
│   ├── quality_analysis.ipynb                # EDA for waste quality data (not included in seminar paper)
│   └── quantity_analysis.ipynb               # EDA for waste quantity data 
│
├── data_preparation/                         
│   ├── __init__.py                           
│   └── data_processor.py                     # Core module with data handling and feature creation logic
│
├── forecasting/                              
│   ├── prophet_predictions/                  
│   │   ├── test_set1_predictions.csv         # Prophet forecast results for comparison (used in DM test)
│   │   └── test_set2_predictions.csv         
│   │
│   ├── xgb_hyperparameters/                  # Best hyperparameters for the XGBoost models
│   │
│   ├── gbt_arrival_time_forecast.ipynb       # XGBoost/CatBoost for arrival time prediction (not included in seminar paper)
│   ├── hyperparameter_tuning.ipynb           # Hyperparameter tuning with Optuna 
│   ├── prophet_quantity_forecast.ipynb       # Facebook Prophet model for waste quantity
│   ├── xgb_xgb_qualityscore_forecast.ipynb   # XGBoost for waste quality score prediction (not included in seminar paper)
│   └── xgb_quantity_forecast.ipynb           # XGBoost models (NAR, AR, ARR), Holt-Winters baseline, DM test
```

## Prerequisites

- **Python Version**: This project was built using Python 3.12.4.

## Installation

Clone this repository and install the required packages:

```bash
git clone https://github.com/1106maxi/AI-supported-Forecasting-Models-for-Waste-Management.git
cd AI-supported-Forecasting-Models-for-Waste-Management
pip install -r requirements.txt
