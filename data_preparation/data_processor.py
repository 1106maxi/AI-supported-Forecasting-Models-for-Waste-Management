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

    # Define holidays as a class attribute
    holidays = [
        '2022-01-01',  # New Year 
        '2022-12-25',  # Christmas
        '2022-04-17',  # Easter      
        '2023-01-01',  # New Year
        '2023-12-25',  # Christmas
        '2023-04-17',  # Easter
        '2024-01-01',  # New Year
        '2024-12-25',  # Christmas
        '2024-04-17'   # Easter
    ]
    holidays = pd.to_datetime(holidays)

    holidays2 = [
        '2022-01-01',  # New Year
        '2022-12-23',  # Christmas time
        '2022-12-24',  # Christmas time
        '2022-12-25',  # Christmas time
        '2022-12-26',  # Christmas time
        '2022-12-27',  # Christmas time
        '2022-12-28',  # Christmas time

        '2023-01-01',  # New Year
        '2023-12-23',  # Christmas time
        '2023-12-24',  # Christmas time
        '2023-12-25',  # Christmas time
        '2023-12-26',  # Christmas time
        '2023-12-27',  # Christmas time
        '2022-12-28',  # Christmas time

        '2024-01-01',  # New Year
        '2024-12-23',  # Christmas time
        '2024-12-24',  # Christmas time
        '2024-12-25',  # Christmas time
        '2024-12-26',  # Christmas time
        '2024-12-27',  # Christmas time
        '2022-12-28',  # Christmas time
        
        '2022-10-03',  # Tag der deutschen Einheit
        '2023-10-03',  # Tag der deutschen Einheit
        '2024-10-03',  # Tag der deutschen Einheit
    ]
    holidays2 = pd.to_datetime(holidays2)



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

    def prepare_quantity_tons(self):
        """
        Prepare the waste data by aggregating quantities, adding seasonal flags, and handling holidays and weekends.

        Returns:
            pd.DataFrame: A DataFrame with aggregated quantities and additional flags.
        """
        df = self.waste_data

        # Convert 'date' to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Group by 'date' and sum 'quantity_tons'
        df_grouped = df.groupby(['date'])['quantity_tons'].sum().reset_index()
        df_grouped = df_grouped.set_index('date').asfreq('D', fill_value=0).reset_index()

        # Add season column
        df['season'] = df['date'].apply(self.get_season)

        # Drop duplicates to ensure one entry per date
        df_unique_dates = df[['date', 'season']].drop_duplicates()

        # Create binary variables for seasons
        season_dummies = pd.get_dummies(df_unique_dates['season'], prefix='is')
        df_season_flags = pd.concat([df_unique_dates['date'], season_dummies], axis=1)

        # Create weekend flag
        df['is_weekend'] = df['date'].dt.weekday.isin([5, 6]).astype(int)

        # Create holiday flag
        df['is_holiday'] = df['date'].isin(self.holidays).astype(int)

        # Aggregate flags
        df_flags = df[['date', 'is_weekend', 'is_holiday']].drop_duplicates().set_index('date')

        # Merge flags with season flags
        df_flags = df_flags.merge(df_season_flags.set_index('date'), on='date', how='outer')

        # Ensure correct data types
        df_flags = df_flags.astype(np.float32)

        # Merge flags with grouped data
        season_df = pd.merge(df_flags, df_grouped, on='date', how='outer')

        return season_df
    
    def create_boosting_features(self, df):
        """
        Creates time series features from datetime index.

        Parameters:
            df (pd.DataFrame): DataFrame for which additional features are added.

        Returns:
            pd.DataFrame: DataFrame with additional features.
        """


        # Ensure the DataFrame has a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

        # Extract time-based features from the datetime index
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['dayofyear'] = df.index.dayofyear
        df['dayofmonth'] = df.index.day
        df['weekofyear'] = df.index.isocalendar().week  # Use isocalendar() for weekofyear

        # Drop the 'date' column (no longer needed)
        if 'date' in df.columns:
            df = df.drop(columns=['date'])

        # Update the waste_data attribute with the new features
        

        return df
    
    def create_lagged_boosting_features(self, df, target_col='quantity_tons', lags=[1, 2, 3, 4, 5, 6]):
        """
        Creates comprehensive time series features for strategic business forecasting.

        Parameters:
            df (pd.DataFrame): DataFrame containing historical business data
            target_col (str): Target column for forecasting
            lags (list): Time lag periods for creating historical reference points
            windows (list): Rolling window sizes for trend and momentum indicators

        Returns:
            pd.DataFrame: Enhanced DataFrame with strategic forecasting features
        """
        # Create a working copy to prevent unintended modifications
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

        # Remove redundant date column if present
        if 'date' in df_copy.columns:
            df_copy = df_copy.drop(columns=['date'])

        # Create lagged features for historical reference points
        for lag in lags:
            df_copy[f'lag_{lag}'] = df_copy[target_col].shift(lag)

        # Generate lag ratio features using sorted lags to ensure consistent progression
        sorted_lags = sorted(lags)
        for i in range(len(sorted_lags)-1):
            current_lag = sorted_lags[i]
            next_lag = sorted_lags[i+1]
            ratio_name = f'lag_ratio_{current_lag}_{next_lag}'
            df_copy[ratio_name] = df_copy[f'lag_{current_lag}'] / df_copy[f'lag_{next_lag}']
            df_copy[ratio_name] = df_copy[ratio_name].replace([np.inf, -np.inf], np.nan)


        # Clean up intermediate lag columns to optimize model input
        for lag in lags:
            df_copy = df_copy.drop(f'lag_{lag}', axis=1)

    

        return df_copy

    def create_lagged_boosting_features2(self, df, target_col='quantity_tons', lags=[1, 2, 3, 4, 5, 6], windows=[7, 14, 30, 90]):
        """
        Creates comprehensive time series features for strategic business forecasting.

        Parameters:
            df (pd.DataFrame): DataFrame containing historical business data
            target_col (str): Target column for forecasting
            lags (list): Time lag periods for creating historical reference points
            windows (list): Rolling window sizes for trend and momentum indicators

        Returns:
            pd.DataFrame: Enhanced DataFrame with strategic forecasting features
        """
        # Create a working copy to prevent unintended modifications
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

        # Remove redundant date column if present
        if 'date' in df_copy.columns:
            df_copy = df_copy.drop(columns=['date'])

        # Create lagged features for historical reference points
        for lag in lags:
            df_copy[f'lag_{lag}'] = df_copy[target_col].shift(lag)

        # Generate lag ratio features using sorted lags to ensure consistent progression
        sorted_lags = sorted(lags)
        for i in range(len(sorted_lags)-1):
            current_lag = sorted_lags[i]
            next_lag = sorted_lags[i+1]
            ratio_name = f'lag_ratio_{current_lag}_{next_lag}'
            df_copy[ratio_name] = df_copy[f'lag_{current_lag}'] / df_copy[f'lag_{next_lag}']
            df_copy[ratio_name] = df_copy[ratio_name].replace([np.inf, -np.inf], np.nan)


        # Implement adaptive trend indicators - Exponentially Weighted Moving Average (EWMA)
        for span in windows:
            df_copy[f'ewma_{span}d'] = df_copy[target_col].shift(1).ewm(span=span).mean()


        # Implement acceleration indicator for short-term directional changes
        df_copy['acceleration_1d'] = df_copy[target_col].shift(1).diff() - df_copy[target_col].shift(2).diff()

        # 7-day acceleration indicator for weekly operational planning
        df_copy['acceleration_7d'] = df_copy[target_col].shift(1).diff(7) - df_copy[target_col].shift(8).diff(7)

        # 90-day acceleration indicator for quarterly strategic planning
        df_copy['acceleration_90d'] = df_copy[target_col].shift(1).diff(90) - df_copy[target_col].shift(91).diff(90)

        # Clean up intermediate lag columns to optimize model input
        for lag in lags:
            df_copy = df_copy.drop(f'lag_{lag}', axis=1)

        # Remove rows with NA values to ensure data integrity
        #df_copy = df_copy.dropna()

        return df_copy



    def prepare_quantity_tons_by_company(self, company):
        """
        Prepare the waste data for a specific company by aggregating quantities, adding seasonal flags,
        and handling holidays and weekends.

        Args:
            company (str): The name of the company to filter the data for.

        Returns:
            pd.DataFrame: A DataFrame with aggregated quantities and additional flags for the specified company.
        """
        # Filter data for the specified company
        df = self.waste_data[self.waste_data['company'] == company].copy()

        # Convert 'date' to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Group by 'date' and sum 'quantity_tons'
        df_grouped = df.groupby(['date'])['quantity_tons'].sum().reset_index()
        df_grouped = df_grouped.set_index('date').asfreq('D', fill_value=0).reset_index()

        # Add season column
        df['season'] = df['date'].apply(self.get_season)

        # Drop duplicates to ensure one entry per date
        df_unique_dates = df[['date', 'season']].drop_duplicates()

        # Create binary variables for seasons
        season_dummies = pd.get_dummies(df_unique_dates['season'], prefix='is')
        df_season_flags = pd.concat([df_unique_dates['date'], season_dummies], axis=1)

        # Create weekend flag
        df['is_weekend'] = df['date'].dt.weekday.isin([5, 6]).astype(int)

        # Create holiday flag
        df['is_holiday'] = df['date'].isin(self.holidays2).astype(int)

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
        season_df = pd.merge(df_flags, df_grouped, on='date', how='outer')

        return season_df
    
    def prepare_quantity_tons_by_waste_type(self, waste_type):
        """
        Prepare the waste data for a specific waste type by aggregating quantities, adding seasonal flags,
        and handling holidays and weekends.
    
        Args:
            waste_type (str): The type of waste to filter the data for.
    
        Returns:
            pd.DataFrame: A DataFrame with aggregated quantities and additional flags for the specified waste type.
        """
        # Filter data for the specified waste type
        df = self.waste_data[self.waste_data['waste_type'] == waste_type].copy()
    
        # Convert 'date' to datetime
        df['date'] = pd.to_datetime(df['date'])
    
        # Group by 'date' and sum 'quantity_tons'
        df_grouped = df.groupby(['date'])['quantity_tons'].sum().reset_index()
        df_grouped = df_grouped.set_index('date').asfreq('D', fill_value=0).reset_index()
    
        # Add season column
        df['season'] = df['date'].apply(self.get_season)
    
        # Drop duplicates to ensure one entry per date
        df_unique_dates = df[['date', 'season']].drop_duplicates()
    
        # Create binary variables for seasons
        season_dummies = pd.get_dummies(df_unique_dates['season'], prefix='is')
        df_season_flags = pd.concat([df_unique_dates['date'], season_dummies], axis=1)
    
        # Create weekend flag
        df['is_weekend'] = df['date'].dt.weekday.isin([5, 6]).astype(int)
    
        # Create holiday flag
        df['is_holiday'] = df['date'].isin(self.holidays).astype(int)
    
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
        season_df = pd.merge(df_flags, df_grouped, on='date', how='outer')
    
        return season_df
    
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