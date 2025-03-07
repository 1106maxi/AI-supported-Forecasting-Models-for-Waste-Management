import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:
    """
    A class to fetch and prepare waste data for analysis.

    Attributes:
        waste_data (pd.DataFrame): The input waste data containing at least 'date' and 'quantity_tons' columns.
        holidays (pd.DatetimeIndex): A list of holidays to be used for flagging.
    """

    # Define base holidays
    holidays_base = [
        '2022-01-01',  # New Year
        '2022-12-23', '2022-12-24', '2022-12-25', '2022-12-26', '2022-12-27', '2022-12-28',  # Christmas time
        '2023-01-01',  # New Year
        '2023-12-23', '2023-12-24', '2023-12-25', '2023-12-26', '2023-12-27', '2023-12-28',  # Christmas time
        '2024-01-01',  # New Year
        '2024-12-23', '2024-12-24', '2024-12-25', '2024-12-26', '2024-12-27', '2024-12-28',  # Christmas time
    ]

    holidays_base = pd.to_datetime(holidays_base)

    all_holidays = list(holidays_base)
    all_holidays = pd.DatetimeIndex(all_holidays).sort_values()


    def __init__(self, waste_data):
        """
        Initialize the FetchData class.

        Args:
            waste_data (pd.DataFrame): The input waste data.
        """
        self.waste_data = waste_data

    @staticmethod
    def get_season(date):
        """
        Determine the season based on the month of the given date.

        Args:
            date (pd.Timestamp): The date to determine the season for.

        Returns:
            str: The season ('Winter', 'Spring', 'Summer', or 'Fall').
        """
        month = date.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'


    def agg_quantity(self, company="", waste_type="", by_company=False, by_waste_type=False):
        """
        Prepare the waste data by aggregating quantities, adding seasonal flags, and handling holidays and weekends.

        Args:
            company (str): The name of the company to filter the data for.
            waste_type (str): The type of waste to filter the data for.
            by_company (bool): Whether to filter by company.
            by_waste_type (bool): Whether to filter by waste type.

        Returns:
            pd.DataFrame: A DataFrame with aggregated quantities and additional flags.
        """
        # Filter data based on the provided criteria
        if by_company:
            df = self.waste_data[self.waste_data['company'] == company].copy()
        elif by_waste_type:
            df = self.waste_data[self.waste_data['waste_type'] == waste_type].copy()
        else:
            df = self.waste_data.copy()

        # Convert 'date' to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Group by 'date' and sum 'quantity_tons', ensuring a complete date range
        df_grouped = df.groupby(['date'])['quantity_tons'].sum().reset_index()
        df_grouped = df_grouped.set_index('date').asfreq('D', fill_value=0).reset_index()

        # Add season column
        df['season'] = df['date'].apply(self.get_season)

        # Drop duplicates to ensure one entry per date
        df_unique_dates = df[['date', 'season']].drop_duplicates()

        # Create binary variables for seasons
        season_dummies = pd.get_dummies(df_unique_dates['season'], prefix='is')
        df_season_flags = pd.concat([df_unique_dates['date'], season_dummies], axis=1)

        # Create weekend & holiday flag
        df['is_weekend'] = df['date'].dt.weekday.isin([5, 6]).astype(int)
        df['is_holiday'] = df['date'].isin(self.all_holidays).astype(int)

        # Aggregate flags
        df_flags = df[['date', 'is_weekend', 'is_holiday']].drop_duplicates().set_index('date')

        # Ensure df_flags includes all dates in the range
        full_date_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
        df_flags = df_flags.reindex(full_date_range).fillna(0).reset_index().rename(columns={'index': 'date'})

        # Merge flags with season flags
        df_flags = df_flags.merge(df_season_flags, on='date', how='left')

        # Fill NaN values in season flags
        season_columns = ['is_Fall', 'is_Spring', 'is_Summer', 'is_Winter']
        df_flags[season_columns] = df_flags[season_columns].fillna(0)

        # Ensure correct data types
        df_flags = df_flags.set_index('date')
        df_flags = df_flags.astype(np.float32)

        # Merge flags with grouped data
        result = pd.merge(df_flags, df_grouped, on='date', how='outer')

        return result
    
    
    def create_xgboost_features(self, df, target_col='quantity_tons', lags=[1], windows=[], lagged_features=True, lagged_ratios=True, trend_indicators=False):
        """
        Creates comprehensive time series features for machine learning models, particularly XGBoost, by incorporating
        historical, seasonal, and trend-based information.  

        Args:
            df (pd.DataFrame): DataFrame containing historical time series data.
            target_col (str): The name of the target column for which features are being created. Default is 'quantity_tons'.
            lags (list): List of integers representing the time lag periods for creating lagged features. Default is [1].
            windows (list): List of integers representing rolling window sizes for trend and momentum indicators. Default is [].
            lagged_features (bool): Whether to create lagged features based on the `lags` parameter. Default is True.
            lagged_ratios (bool): Whether to create ratio features between consecutive lagged values. Default is True.
            trend_indicators (bool): Whether to create trend-based features such as exponentially weighted moving averages
                                    and acceleration indicators. Default is False.  

        Returns:
            pd.DataFrame: Enhanced DataFrame with additional time series features for forecasting.
        """

        df_copy = df.copy() 

        # Standardize datetime indexing for consistent time series analysis
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            df_copy = df_copy.set_index('date') 

        # Extract calendar-based features for seasonal pattern identification
        df_copy['dayofweek'] = df_copy.index.dayofweek
        df_copy['quarter'] = df_copy.index.quarter
        df_copy['month'] = df_copy.index.month
        df_copy['year'] = df_copy.index.year
        df_copy['dayofyear'] = df_copy.index.dayofyear
        df_copy['dayofmonth'] = df_copy.index.day
        df_copy['weekofyear'] = df_copy.index.isocalendar().week    

        if 'date' in df_copy.columns:
            df_copy = df_copy.drop(columns=['date'])    

        # Create lagged features for historical reference points
        for lag in lags:
            df_copy[f'lag_{lag}'] = df_copy[target_col].shift(lag)  

        # Generate lag ratio features using sorted lags to ensure consistent progression
        if lagged_ratios == True:
            sorted_lags = sorted(lags)
            for i in range(len(sorted_lags)-1):
                current_lag = sorted_lags[i]
                next_lag = sorted_lags[i+1]
                ratio_name = f'lag_ratio_{current_lag}_{next_lag}'
                df_copy[ratio_name] = df_copy[f'lag_{current_lag}'] / df_copy[f'lag_{next_lag}']
                df_copy[ratio_name] = df_copy[ratio_name].replace([np.inf, -np.inf], np.nan)    
        
        # Drop rows where lagged features or lagged ratios are NaN
        if lagged_features or lagged_ratios:
        # Columns to check for NaN values
            columns_to_check = [f'lag_{lag}' for lag in lags] if lagged_features else []
            if lagged_ratios:
                columns_to_check += [f'lag_ratio_{sorted_lags[i]}_{sorted_lags[i + 1]}' for i in range(len(sorted_lags) - 1)]
        
        # Drop rows with NaN values in any of the specified columns
        df_copy = df_copy.dropna(subset=columns_to_check)

        # Drop lagged features if not needed
        if lagged_features == False:
            for lag in lags:
                df_copy = df_copy.drop(f'lag_{lag}', axis=1)    

        # Create trend-based features if enabled
        if trend_indicators == True:
            # Implement adaptive trend indicators - Exponentially Weighted Moving Average (EWMA)
            for span in windows:
                df_copy[f'ewma_{span}d'] = df_copy[target_col].shift(1).ewm(span=span).mean()   

            # Implement acceleration indicator for short-term directional changes
            df_copy['acceleration_3d'] = df_copy[target_col].shift(1).diff(3) - df_copy[target_col].shift(2).diff(3) 
 
        
        return df_copy


    
    def gru_prepare_quantity_tons_by_waste_type(self, waste_type, timesteps=7):
        """
        Prepare the data for a GRU model by creating sequences and normalizing the data.

        Args:
            waste_type (str): The type of waste to filter the data for.
            timesteps (int): The number of timesteps for the GRU model.

        Returns:
            tuple: A tuple containing:
                - X_train_gru (np.array): Training sequences.
                - y_train_gru (np.array): Training targets.
                - X_test_gru (np.array): Test sequences.
                - y_test_gru (np.array): Test targets.
                - scaler (MinMaxScaler): The scaler used to normalize the data.
        """
        # Filter data for the specified waste type
        df = self.waste_data[self.waste_data['waste_type'] == waste_type].copy()

        # Convert 'date' to datetime and sort
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Extract the target variable
        y = df['quantity_tons'].values

        # Train-test split
        split_index = int(len(df) * 0.8)
        y_train, y_test = y[:split_index], y[split_index:]

        # Normalize the data
        scaler = MinMaxScaler()
        y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
        y_test_scaled = scaler.transform(y_test.reshape(-1, 1))

        # Create sequences
        def create_sequences(data, timesteps):
            X, y = [], []
            for i in range(len(data) - timesteps):
                X.append(data[i:i + timesteps])
                y.append(data[i + timesteps])
            return np.array(X), np.array(y)

        X_train_gru, y_train_gru = create_sequences(y_train_scaled, timesteps)
        X_test_gru, y_test_gru = create_sequences(y_test_scaled, timesteps)

        return X_train_gru, y_train_gru, X_test_gru, y_test_gru, scaler