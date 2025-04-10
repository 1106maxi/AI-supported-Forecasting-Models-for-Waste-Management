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


    def agg_quantity(self, company="", waste_type="", by_company=False, by_waste_type=False, one_hot=True):
        """
        Prepare the waste data by aggregating quantities, adding seasonal flags, and handling holidays and weekends.
    
        Args:
            company (str): The name of the company to filter the data for.
            waste_type (str): The type of waste to filter the data for.
            by_company (bool): Whether to filter by company.
            by_waste_type (bool): Whether to filter by waste type.
            one_hot (bool): If True, season and holiday features are one-hot encoded.
                             If False, they are kept as single categorical features with names.
        
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
    
        # Creating a skeleton df to fetch weekend, season and holiday flags from. 
        # This is done because when aggregating by company some dates are missing in the df_grouped dataframe
        # because not every company delivers every day.
        date_range = pd.date_range(start='2022-01-01', end='2024-12-31', freq='d') 
        skeleton_df = pd.DataFrame(date_range, columns=["date"])

        # Convert 'date' to datetime
        df['date'] = pd.to_datetime(df['date'])
    
        # Group by 'date' and sum 'quantity_tons', ensuring a complete date range
        df_grouped = df.groupby(['date'])['quantity_tons'].sum().reset_index()
        df_grouped = df_grouped.set_index('date').asfreq('D', fill_value=0).reset_index()
    
        # Add season column using the helper function
        skeleton_df["season"] = skeleton_df['date'].apply(self.get_season)
        #df_grouped['season'] = skeleton_df['date'].apply(self.get_season)
        df_grouped = pd.merge(skeleton_df, df_grouped, on = "date", how = "left")

        if one_hot:
            # --- ONE-HOT ENCODING ---

            # Create one entry per date for season using one-hot encoding
            season_dummies = pd.get_dummies(df_grouped['season'], prefix='is')
            df_season_flags = pd.concat([df_grouped['date'], season_dummies], axis=1)                      

            # Create weekend flag (binary: 1 if weekend, else 0)
            df_grouped['is_weekend'] = skeleton_df['date'].dt.weekday.isin([5, 6]).astype(int)
            # Create holiday flag (binary: 1 if the date is in all_holidays, else 0)
            df_grouped['is_holiday'] = skeleton_df['date'].isin(self.all_holidays).astype(int)
    
            # Merge with season one-hot flags
            df_grouped = df_grouped.drop("season", axis = 1) 
            df_flags = pd.merge(df_season_flags,df_grouped , on='date', how='inner')

            # Ensure correct index and data types
            df_flags = df_flags.set_index('date')
            df_flags = df_flags.fillna(0)
            df_flags = df_flags.astype(np.float32)
            
            result = df_flags

        else:
            # --- CATEGORICAL FEATURES ---
    
            # Define a helper function to get holiday names
            def get_holiday_name(date):
                # If date is a holiday, assign name based on month-day
                if date in self.all_holidays:
                    if date.strftime('%m-%d') == '01-01':
                        return 'New Year'
                    elif date.strftime('%m-%d') in ['12-23', '12-24', '12-25', '12-26', '12-27', '12-28']:
                        return 'Christmas time'
                    else:
                        return 'Holiday'
                else:
                    return 'No Holiday'
    
            # Create a DataFrame with weekend flag and holiday name
            df_weekend = skeleton_df[['date']].copy()
            df_weekend['is_weekend'] = df_weekend['date'].dt.weekday.isin([5, 6]).astype(int)
            df_weekend['holiday'] = df_weekend['date'].apply(get_holiday_name)
            df_weekend['holiday'] = df_weekend['holiday'].fillna('No Holiday')
    
            # Merge the season (categorical) with holiday and weekend info
            df_flags = pd.merge(df_weekend, df_grouped, on='date', how='left')

            # Ensure correct index
            df_flags = df_flags.fillna(0)
            df_flags = df_flags.set_index('date')
    
        result = df_flags

        return result

    
    
    def create_xgboost_features(self, df, waste_type = "" ,target_col='quantity_tons', lags=[1], windows=[], lagged_features=True, lagged_ratios=True, trend_indicators=False, fourier_terms=False, interaction_terms=False, trend_term=False):
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
            fourier_terms (bool): Whether to create Fourier terms for capturing cyclical patterns. Default is False.
            interaction_terms (bool): Whether to create interaction terms between existing features. Default is False.
            trend_term (bool): Whether to include a linear trend term. Default is False.
    
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
            columns_to_check = [f'lag_{lag}' for lag in lags] if lagged_features else []
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
    
        # Create Fourier terms for capturing cyclical patterns
        if fourier_terms == True:
            annual_period = 365.33
            weekly_period = 7.0
            semi_weekly_period = 3.50
            industrial_period = 2
            time = np.arange(len(df_copy))

            if waste_type == 'Municipal':
                # Annual Fourier terms
                for i in range(1, 7):
                    df_copy[f'annual_fourier_sin_{i}'] = np.sin(2 * np.pi * i * time / annual_period)
                    df_copy[f'annual_fourier_cos_{i}'] = np.cos(2 * np.pi * i * time / annual_period)

                # Weekly Fourier terms
                for i in range(1, 3):
                    df_copy[f'weekly_fourier_sin_{i}'] = np.sin(2 * np.pi * i * time / weekly_period)
                    df_copy[f'weekly_fourier_cos_{i}'] = np.cos(2 * np.pi * i * time / weekly_period)

                # Semi-weekly Fourier terms
                for i in range(1, 3):
                    df_copy[f'semi_weekly_fourier_sin_{i}'] = np.sin(2 * np.pi * i * time / semi_weekly_period)
                    df_copy[f'semi_weekly_fourier_cos_{i}'] = np.cos(2 * np.pi * i * time / semi_weekly_period)

            elif waste_type == 'Industrial':
                # Industrial-specific Fourier terms
                for i in range(1, 3):
                    df_copy[f'industrial_fourier_sin_{i}'] = np.sin(2 * np.pi * i * time / industrial_period)
                    df_copy[f'industrial_fourier_cos_{i}'] = np.cos(2 * np.pi * i * time / industrial_period)

                # Weekly Fourier terms
                df_copy[f'weekly_fourier_sin_{i}'] = np.sin(2 * np.pi * i * time / weekly_period)
                df_copy[f'weekly_fourier_cos_{i}'] = np.cos(2 * np.pi * i * time / weekly_period)

            elif waste_type == 'Organic':
                # Annual Fourier terms
                for i in range(1, 7):
                    df_copy[f'annual_fourier_sin_{i}'] = np.sin(2 * np.pi * i * time / annual_period)
                    df_copy[f'annual_fourier_cos_{i}'] = np.cos(2 * np.pi * i * time / annual_period)

            elif waste_type == 'Construction':
                # Annual Fourier terms
                for i in range(1, 7):
                    df_copy[f'annual_fourier_sin_{i}'] = np.sin(2 * np.pi * i * time / annual_period)
                    df_copy[f'annual_fourier_cos_{i}'] = np.cos(2 * np.pi * i * time / annual_period)

                # Semi-weekly Fourier terms
                for i in range(1, 3):
                    df_copy[f'semi_weekly_fourier_sin_{i}'] = np.sin(2 * np.pi * i * time / semi_weekly_period)
                    df_copy[f'semi_weekly_fourier_cos_{i}'] = np.cos(2 * np.pi * i * time / semi_weekly_period)

            elif waste_type == 'Commercial':
                # Weekly Fourier terms
                for i in range(1, 3):
                    df_copy[f'weekly_fourier_sin_{i}'] = np.sin(2 * np.pi * i * time / weekly_period)
                    df_copy[f'weekly_fourier_cos_{i}'] = np.cos(2 * np.pi * i * time / weekly_period)

                # Semi-weekly Fourier terms
                for i in range(1, 3):
                    df_copy[f'semi_weekly_fourier_sin_{i}'] = np.sin(2 * np.pi * i * time / semi_weekly_period)
                    df_copy[f'semi_weekly_fourier_cos_{i}'] = np.cos(2 * np.pi * i * time / semi_weekly_period)

        # Create interaction terms between existing features
        if interaction_terms == True:
            df_copy['month_dayofweek'] = df_copy['month'] * df_copy['dayofweek']
            df_copy['quarter_dayofweek'] = df_copy['quarter'] * df_copy['dayofweek']
        # Include a linear trend term
        if trend_term == True:
            df_copy['trend'] = np.arange(len(df_copy))
            
        return df_copy 

    def arriavl_time(self, company = "", by_company = False, agg_arrival = False, agg_value = ""):
        """
        
        """
        df = self.waste_data.copy()
        df = df.drop(["quantity_tons", "quality_score", "moisture_content", "contamination_level", "heating_value_MJ_per_kg"], axis = 1)


        if by_company == True:
            target_df = df[df['company'] == company].reset_index()
            target_df = target_df.drop("index",axis = 1)
        else:
            target_df = df.copy()

        target_df = target_df.drop(["date", "company", "waste_type", "truck_id"], axis = 1)
        target_df["arrival"] = 1
        
        # create skeleton df
        date_range = pd.date_range(start='2022-01-01', end='2024-12-31', freq='min') 
        skeleton_df = pd.DataFrame(date_range, columns=['arrival_time'])

        target_df["arrival_time"] = pd.to_datetime(target_df["arrival_time"])
        minute_df = pd.merge(skeleton_df, target_df, on='arrival_time', how='left')
        minute_df.fillna(0, inplace=True)
        minute_df['season'] = minute_df['arrival_time'].apply(self.get_season)

        season_dummies = pd.get_dummies(minute_df['season'], prefix='is')
        minute_df = pd.concat([minute_df, season_dummies], axis=1)
        minute_df['is_weekend'] = minute_df['arrival_time'].dt.weekday.isin([5, 6]).astype(int)
        minute_df['is_holiday'] = minute_df['arrival_time'].isin(self.all_holidays).astype(int)
        final_df = minute_df.drop("season", axis = 1)

        final_df['dayofweek'] = final_df['arrival_time'].dt.dayofweek
        final_df['quarter'] = final_df['arrival_time'].dt.quarter
        final_df['month'] = final_df['arrival_time'].dt.month
        final_df['year'] = final_df['arrival_time'].dt.year
        final_df['dayofyear'] = final_df['arrival_time'].dt.dayofyear
        final_df['dayofmonth'] = final_df['arrival_time'].dt.day
        final_df['weekofyear'] = final_df['arrival_time'].dt.isocalendar().week 

        final_df['hour'] = final_df['arrival_time'].dt.hour
        final_df['day_of_week'] = final_df['arrival_time'].dt.dayofweek
        final_df['month'] = final_df['arrival_time'].dt.month
    

        final_df = final_df.set_index("arrival_time")

        if agg_arrival == True:
            final_df = final_df.resample(f'{agg_value}h').agg({

            'is_Fall': 'max',
            'is_Spring': 'max',
            'is_Summer': 'max', 
            'is_Winter': 'max',
            'is_weekend': 'max',
            'is_holiday': 'max',
            "arrival": "max",

            'dayofweek': 'first',
            'quarter': 'first',
            'month': 'first',
            'year': 'first',
            'dayofyear': 'first',
            'dayofmonth': 'first',
            'weekofyear': 'first',
            'hour': lambda x: (x.iloc[0] // agg_value) * agg_value, 
            'day_of_week': 'first'})

        final_df = final_df.astype(np.float32)

        return final_df

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