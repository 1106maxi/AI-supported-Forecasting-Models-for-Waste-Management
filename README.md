# AI-supported Forecasting Models for Waste Management and Resource Planning

This seminar topic focuses on developing a forecasting system for optimizing waste processing in
waste incineration plants.
• Prediction of arrival times, quantities, and qualities
• Consideration of seasonal fluctuations and regional characteristics
• Automatic detection of relevant influencing factors
The developed system should enable more efficient resource planning in disposal facilities
and contribute to improving processing procedures. The implementation should be flexible to
evaluate various forecasting models and examine the effects of different parameters on prediction
accuracy. 

## Notebooks

### 1. `data_examination_quantity`
This notebook provides a **descriptive analysis** of the quantity of delivered waste. By exploring trends, seasonality, and key statistics, we gain valuable insights into waste delivery patterns, which serve as the foundation for building predictive models.

**Key Highlights**:
- Exploratory Data Analysis of waste quantity data.
- Identification of key patterns to inform feature engineering and model development.

---

### 2. `forecasting_xgboost_quantity`
This notebook focuses on **forecasting waste delivery quantities** using XGBoost, a powerful machine learning algorithm. By experimenting with different features—such as lagged variables, rolling statistics, and seasonal indicators—we aim to build accurate and interpretable models for waste management and resource planning.

**Key Highlights**:
- Feature engineering tailored for time series data.
- Hyperparameter tuning to optimize model performance.
- Evaluation of forecasting accuracy and interpretation of results.

---

### 3. `test`
currently under construction 