import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import subprocess
import importlib
import warnings
import os
import json
from datetime import datetime, timedelta
import logging
warnings.filterwarnings('ignore')
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('wind_power_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# Try importing sklearn symbols at module import time
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except Exception:
    RandomForestRegressor = None
    GradientBoostingRegressor = None
    LinearRegression = None
    StandardScaler = None
    mean_absolute_error = None
    mean_squared_error = None
    r2_score = None
class WindPowerPredictor:
    """
    Cloud-Based Smart Scheduling System for Renewable Power Generation
    Now supports:
    - Full year prediction
    - Specific day prediction
    - Date range prediction
    - Specific timestamp prediction (NEW!)
    """
    COLUMN_MAP = {
        'Wind Speed at 78.5 mtr': 'Wind Speed (m/s)',
        'Wind Direction at 78.5 mtr': 'Wind Direction (degrees)',
        'Ambient Temp at 78.5 mtr': 'Ambient Temperature (Â°C)',
        'Active_Power 78.5 mtr': 'Power Generated (kW)',
    }
    REQUIRED_COLUMNS = list(COLUMN_MAP.values()) + ['Rotor RPM']
    def __init__(self, base_path, train_files, test_file, output_dir='output'):
        self.base_path = base_path
        self.train_files = train_files
        self.test_file = test_file
        self.output_dir = output_dir
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.train_data = None
        self.test_data = None
        self.execution_metadata = {
            'start_time': datetime.now().isoformat(),
            'model_type': None,
            'train_samples': 0,
            'test_samples': 0
        }
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initialized WindPowerPredictor with output directory: {output_dir}")
        try:
            self._ensure_ml_packages()
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            globals().update({
                'RandomForestRegressor': RandomForestRegressor,
                'GradientBoostingRegressor': GradientBoostingRegressor,
                'LinearRegression': LinearRegression,
                'StandardScaler': StandardScaler,
                'mean_absolute_error': mean_absolute_error,
                'mean_squared_error': mean_squared_error,
                'r2_score': r2_score
            })

            self.scaler = StandardScaler()
        except Exception as e:
            logger.error(f"Failed to ensure ML packages or import sklearn: {e}")
            raise
    def _ensure_ml_packages(self):
        try:
            import sklearn
            return
        except Exception:
            logger.info("scikit-learn not found; attempting installation via pip...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy', 'scipy', 'scikit-learn'])
            importlib.invalidate_caches()
            logger.info("Successfully installed scikit-learn and dependencies.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Automatic installation failed: {e}")
            raise RuntimeError("Failed to install scikit-learn automatically.")
    def _load_and_clean_scada(self, file_path, file_name):
        try:
            logger.info(f"Loading SCADA data from: {file_name}")
            df = pd.read_csv(file_path, header=[3], index_col=0)
        except Exception as e:
            logger.error(f"Failed to read CSV: {file_path}")
            raise Exception(f"Failed to read CSV at {file_path}. Error: {e}")
        df.columns = df.columns.str.strip()
        df = df.iloc[1:].copy()
        df.index.name = 'Timestamp'
        df = df.reset_index()
        measurement_cols = [col for col in df.columns if col in self.COLUMN_MAP]
        if not measurement_cols:
            raise KeyError(f"No measurement columns found in {file_name}.")
        df_melt = df[['Timestamp']].copy()
        num_params_per_turbine = len(self.COLUMN_MAP)
        for i, col_name in enumerate(measurement_cols):
            turbine_idx = i // num_params_per_turbine + 1
            new_col_name = f"T{turbine_idx}_{col_name}"
            df_melt[new_col_name] = pd.to_numeric(df.iloc[:, i + 1], errors='coerce')
        df_long = pd.melt(df_melt, id_vars=['Timestamp'],
                          var_name='Turbine_Parameter', value_name='Value')
        df_long[['Turbine_ID', 'Parameter']] = df_long['Turbine_Parameter'].str.split('_', n=1, expand=True)
        df_final = df_long.pivot_table(
            index=['Timestamp', 'Turbine_ID'],
            columns='Parameter',
            values='Value'
        ).reset_index()
        df_final.rename(columns=self.COLUMN_MAP, inplace=True)
        df_final['Rotor RPM'] = df_final['Wind Speed (m/s)'] * 15 + np.random.normal(0, 5, len(df_final))
        df_final['Rotor RPM'] = np.clip(df_final['Rotor RPM'], 0, 300)
        df_final.dropna(inplace=True)
        logger.info(f"Successfully loaded {len(df_final)} records from {file_name}")
        return df_final
    def load_and_prepare_data(self):
        logger.info("Starting data loading and preparation...")
        all_dfs = []
        for file_name in self.train_files + [self.test_file]:
            file_path = os.path.join(self.base_path, file_name)
            try:
                df = self._load_and_clean_scada(file_path, file_name)
                df['Source_File'] = file_name
                all_dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to process {file_name}: {e}")
                raise
        df = pd.concat(all_dfs, ignore_index=True)
        df['Timestamp'] = self._parse_timestamps(df['Timestamp'])
        df = df.sort_values('Timestamp').reset_index(drop=True)
        df = df.set_index('Timestamp')
        df = df.asfreq('10T', method='ffill')
        df = df.reset_index()

        logger.info(f"Total records loaded: {len(df)}")
        return df
    def _parse_timestamps(self, series):
        s = series.astype(str).str.strip()
        dt = pd.to_datetime(s, errors='coerce', infer_datetime_format=True)
        if dt.isna().any():
            dt_dayfirst = pd.to_datetime(s, errors='coerce', dayfirst=True, infer_datetime_format=True)
            dt = dt.fillna(dt_dayfirst)
        if dt.isna().any():
            formats = [
                "%d-%m-%Y %H:%M",
                "%m-%d-%Y %H:%M",
                "%Y-%m-%d %H:%M",
                "%d-%m-%Y %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y/%m/%d %H:%M"
            ]
            for fmt in formats:
                parsed = pd.to_datetime(s, format=fmt, errors='coerce')
                dt = dt.fillna(parsed)
        if dt.isna().any():
            try:
                from dateutil.parser import parse
                mask = dt.isna()
                values = s[mask].tolist()
                parsed_vals = []
                for v in values:
                    try:
                        parsed = parse(v, dayfirst=True)
                    except:
                        try:
                            parsed = parse(v, dayfirst=False)
                        except:
                            parsed = pd.NaT
                    parsed_vals.append(parsed)
                dt.loc[mask] = parsed_vals
            except:
                pass
        nat_count = int(dt.isna().sum())
        if nat_count > 0:
            logger.warning(f"{nat_count} timestamps could not be parsed.")
        return dt
    def create_features(self, df):
        logger.info("Creating engineered features...")
        df = df.copy()

        if 'Wind Direction (degrees)' in df.columns:
            df['wind_x'] = np.sin(np.radians(df['Wind Direction (degrees)']))
            df['wind_y'] = np.cos(np.radians(df['Wind Direction (degrees)']))
        else:
            df['wind_x'] = 0
            df['wind_y'] = 0
        df['hour'] = df['Timestamp'].dt.hour
        df['day_of_year'] = df['Timestamp'].dt.dayofyear
        df['month'] = df['Timestamp'].dt.month
        df['season'] = (df['month'] % 12 + 3) // 3
        df['day_of_week'] = df['Timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['wind_speed_lag1'] = df['Wind Speed (m/s)'].shift(1)
        df['wind_speed_lag6'] = df['Wind Speed (m/s)'].shift(6)
        df['power_lag1'] = df['Power Generated (kW)'].shift(1)
        df['power_lag6'] = df['Power Generated (kW)'].shift(6)
        df['wind_speed_roll6'] = df['Wind Speed (m/s)'].rolling(window=6, min_periods=1).mean()
        df['wind_speed_roll12'] = df['Wind Speed (m/s)'].rolling(window=12, min_periods=1).mean()
        df['wind_speed_roll24'] = df['Wind Speed (m/s)'].rolling(window=24, min_periods=1).mean()
        df['power_roll6'] = df['Power Generated (kW)'].rolling(window=6, min_periods=1).mean()
        df['power_roll12'] = df['Power Generated (kW)'].rolling(window=12, min_periods=1).mean()
        df['wind_speed_std6'] = df['Wind Speed (m/s)'].rolling(window=6, min_periods=1).std()
        df['temp_roll6'] = df['Ambient Temperature (Â°C)'].rolling(window=6, min_periods=1).mean()
        df['wind_temp_interaction'] = df['Wind Speed (m/s)'] * df['Ambient Temperature (Â°C)']
        df['wind_speed_squared'] = df['Wind Speed (m/s)'] ** 2
        df['wind_speed_cubed'] = df['Wind Speed (m/s)'] ** 3
        df = df.dropna()
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df
    def split_train_test(self, df, specific_date=None, date_range=None):
        """
        Split data - now supports specific date filtering
        Args:
            df: Combined DataFrame
            specific_date: Single date string 'YYYY-MM-DD' for single day prediction
            date_range: Tuple of ('YYYY-MM-DD', 'YYYY-MM-DD') for range prediction
        """
        logger.info("Splitting data into train and test sets...")
        train_df = df[df['Source_File'].isin(self.train_files)].copy()
        test_df = df[df['Source_File'] == self.test_file].copy()
        # Filter test data by specific date or date range
        if specific_date:
            target_date = pd.to_datetime(specific_date)
            test_df = test_df[test_df['Timestamp'].dt.date == target_date.date()].copy()
            logger.info(f"Filtered test data for specific date: {specific_date}")
        elif date_range:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            test_df = test_df[
                (test_df['Timestamp'].dt.date >= start_date.date()) &
                (test_df['Timestamp'].dt.date <= end_date.date())
            ].copy()
            logger.info(f"Filtered test data for date range: {date_range[0]} to {date_range[1]}")
        train_df = train_df.drop(columns=['Source_File'])
        test_df = test_df.drop(columns=['Source_File'])
        logger.info(f"Train: {len(train_df):,} samples | Test: {len(test_df):,} samples")
        self.execution_metadata['train_samples'] = len(train_df)
        self.execution_metadata['test_samples'] = len(test_df)
        if len(test_df) == 0:
            raise ValueError(f"No data found for the specified date/range")
        return train_df, test_df
    def prepare_features(self, df):
        feature_cols = [
            'Wind Speed (m/s)', 'wind_x', 'wind_y',
            'Ambient Temperature (Â°C)', 'Rotor RPM',
            'hour', 'day_of_year', 'season', 'day_of_week', 'is_weekend',
            'wind_speed_lag1', 'wind_speed_lag6',
            'power_lag1', 'power_lag6',
            'wind_speed_roll6', 'wind_speed_roll12', 'wind_speed_roll24',
            'power_roll6', 'power_roll12',
            'wind_speed_std6', 'temp_roll6',
            'wind_temp_interaction', 'wind_speed_squared', 'wind_speed_cubed',
            'Turbine_ID'
        ]
        df = pd.get_dummies(df, columns=['Turbine_ID'], prefix='TID', drop_first=True)
        feature_cols_final = [col for col in feature_cols if col != 'Turbine_ID']
        feature_cols_final += [col for col in df.columns if col.startswith('TID_')]
        X = df[feature_cols_final].copy()
        y = df['Power Generated (kW)'].copy()
        return X, y, feature_cols_final
    def train_model(self, model_type='random_forest'):
        logger.info(f"Training {model_type} model...")
        self.execution_metadata['model_type'] = model_type

        X_train, y_train, self.feature_columns = self.prepare_features(self.train_data)
        X_train_scaled = self.scaler.fit_transform(X_train)
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError("Model type must be 'linear', 'random_forest', or 'gradient_boosting'")
        self.model.fit(X_train_scaled, y_train)
        train_score = self.model.score(X_train_scaled, y_train)
        logger.info(f"Training complete. R² score: {train_score:.4f}")
        self.execution_metadata['train_r2'] = train_score
        return self.model
    def predict(self):
        logger.info("Making predictions on test data...")
        X_test, y_test, features = self.prepare_features(self.test_data)
        missing_cols = set(self.feature_columns) - set(X_test.columns)
        for c in missing_cols:
            X_test[c] = 0
        X_test = X_test[self.feature_columns]
        X_test_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_test_scaled)
        predictions = np.maximum(predictions, 0)
        logger.info(f"Predictions generated for {len(predictions)} samples")
        return predictions, y_test.values
    def predict_specific_timestamp(self, date_str, time_str):
        """
        Predict power generation for a specific date and time

        Args:
            date_str: Date in 'DD-MM-YYYY' or 'YYYY-MM-DD' format
            time_str: Time in 'HH:MM' format
        Returns:
            Dictionary with predictions for all turbines
        """
        try:
            # Parse date and time
            try:
                target_dt = pd.to_datetime(f"{date_str} {time_str}", dayfirst=True)
            except:
                target_dt = pd.to_datetime(f"{date_str} {time_str}")
            logger.info(f"Predicting for timestamp: {target_dt}")
            # Find nearest timestamp in test data (within 10-minute intervals)
            self.test_data['time_diff'] = abs((self.test_data['Timestamp'] - target_dt).dt.total_seconds())
            closest_idx = self.test_data['time_diff'].idxmin()
            closest_timestamp = self.test_data.loc[closest_idx, 'Timestamp']
            if self.test_data.loc[closest_idx, 'time_diff'] > 600:  # More than 10 minutes
                logger.warning(f"No exact match. Closest available: {closest_timestamp}")
            else:
                logger.info(f"Found matching timestamp: {closest_timestamp}")

            # Get all turbine data for this timestamp
            timestamp_data = self.test_data[
                self.test_data['Timestamp'] == closest_timestamp
            ].copy()
            if len(timestamp_data) == 0:
                raise ValueError(f"No turbine data found for timestamp: {closest_timestamp}")
            logger.info(f"Found {len(timestamp_data)} turbines for prediction")
            # Store original turbine IDs before one-hot encoding
            original_turbine_ids = timestamp_data['Turbine_ID'].values.copy()
            # One-hot encode Turbine_ID
            timestamp_data_encoded = pd.get_dummies(
                timestamp_data,
                columns=['Turbine_ID'],
                prefix='TID',
                drop_first=True
            )
            # Prepare feature columns (excluding target variable)
            feature_cols = [
                'Wind Speed (m/s)', 'wind_x', 'wind_y',
                'Ambient Temperature (Â°C)', 'Rotor RPM',
                'hour', 'day_of_year', 'season', 'day_of_week', 'is_weekend',
                'wind_speed_lag1', 'wind_speed_lag6',
                'power_lag1', 'power_lag6',
                'wind_speed_roll6', 'wind_speed_roll12', 'wind_speed_roll24',
                'power_roll6', 'power_roll12',
                'wind_speed_std6', 'temp_roll6',
                'wind_temp_interaction', 'wind_speed_squared', 'wind_speed_cubed'
            ]
            # Add TID columns
            tid_cols = [col for col in timestamp_data_encoded.columns if col.startswith('TID_')]
            all_features = feature_cols + tid_cols
            # Get features that exist in the data
            available_features = [col for col in all_features if col in timestamp_data_encoded.columns]
            X_test = timestamp_data_encoded[available_features].copy()
            y_test = timestamp_data_encoded['Power Generated (kW)'].values
            # Add missing feature columns from training (set to 0)
            for col in self.feature_columns:
                if col not in X_test.columns:
                    X_test[col] = 0
            # Ensure column order matches training
            X_test = X_test[self.feature_columns]
            # Scale features
            X_test_scaled = self.scaler.transform(X_test)
            # Predict
            predictions = self.model.predict(X_test_scaled)
            predictions = np.maximum(predictions, 0)
            logger.info(f"Generated {len(predictions)} predictions")
            # Create results with original turbine IDs
            results = {
                'timestamp': closest_timestamp,
                'predictions': {},
                'actual': {},
                'total_predicted': 0,
                'total_actual': 0
            }
            for idx, turbine_id in enumerate(original_turbine_ids):
                pred_val = float(predictions[idx])
                actual_val = float(y_test[idx])
                results['predictions'][turbine_id] = pred_val
                results['actual'][turbine_id] = actual_val
                results['total_predicted'] += pred_val
                results['total_actual'] += actual_val
            logger.info(f"Total predicted: {results['total_predicted']:.2f} kW")
            logger.info(f"Total actual: {results['total_actual']:.2f} kW")

            return results
        except Exception as e:
            logger.error(f"Failed to predict for specific timestamp: {e}", exc_info=True)
            raise
    def display_timestamp_prediction(self, results):
        """Display formatted results for timestamp prediction"""
        print("\n" + "="*70)
        print("SPECIFIC TIMESTAMP PREDICTION")
        print("="*70)
        print(f"\nTimestamp: {results['timestamp']}")
        print(f"Turbines analyzed: {len(results['predictions'])}")
        print("\n" + "-"*70)
        print("TURBINE-WISE PREDICTIONS")
        print("-"*70)
        print(f"{'Turbine ID':<15} {'Predicted (kW)':<20} {'Actual (kW)':<20} {'Error (kW)':<15}")
        print("-"*70)
        for turbine_id in sorted(results['predictions'].keys()):
            pred = results['predictions'][turbine_id]
            actual = results['actual'][turbine_id]
            error = abs(pred - actual)
            print(f"{turbine_id:<15} {pred:>18.2f}  {actual:>18.2f}  {error:>13.2f}")

        print("-"*70)
        print(f"{'TOTAL':<15} {results['total_predicted']:>18.2f}  {results['total_actual']:>18.2f}  {abs(results['total_predicted']-results['total_actual']):>13.2f}")
        print("\n" + "-"*70)
        print("SUMMARY STATISTICS")
        print("-"*70)
        print(f"Total Predicted Power:  {results['total_predicted']:>15.2f} kW")
        print(f"Total Actual Power:     {results['total_actual']:>15.2f} kW")
        print(f"Total Deviation:        {abs(results['total_predicted']-results['total_actual']):>15.2f} kW")
        if results['total_actual'] > 0:
            accuracy = 100 - (abs(results['total_predicted']-results['total_actual'])/results['total_actual']*100)
            print(f"Prediction Accuracy:    {accuracy:>15.2f}%")
        print("="*70 + "\n")
    def calculate_errors(self, actual, predicted):
        logger.info("Calculating error metrics...")
        error = actual - predicted
        absolute_error = np.abs(error)
        percentage_error = np.where(actual != 0, (absolute_error / actual) * 100, 0)
        percentage_error = np.clip(percentage_error, 0, 100)
        results_df = pd.DataFrame({
            'Timestamp': self.test_data['Timestamp'].values[:len(actual)],
            'Turbine_ID': self.test_data['Turbine_ID'].values[:len(actual)],
            'Actual': actual,
            'Predicted': predicted,
            'Error': error,
            'Absolute_Error': absolute_error,
            'Percentage_Error': percentage_error,
            'Squared_Error': error ** 2
        })
        results_df['Hour'] = pd.to_datetime(results_df['Timestamp']).dt.hour
        results_df['Month'] = pd.to_datetime(results_df['Timestamp']).dt.month
        results_df['Season'] = ((results_df['Month'] % 12 + 3) // 3)
        return results_df
    def calculate_metrics(self, actual, predicted):
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100
        max_error = np.max(np.abs(actual - predicted))
        median_ae = np.median(np.abs(actual - predicted))
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Max_Error': max_error,
            'Median_AE': median_ae,
            'Mean_Actual': np.mean(actual),
            'Mean_Predicted': np.mean(predicted)
        }
        logger.info(f"Performance Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
        return metrics
    def generate_daily_summary(self, results_df):
        """Generate summary for specific day prediction"""
        print("\n" + "="*70)
        print("DAILY PREDICTION SUMMARY")
        print("="*70)
        date_str = results_df['Timestamp'].iloc[0].strftime('%Y-%m-%d')
        print(f"\nDate: {date_str}")
        print(f"Total Intervals: {len(results_df)} (10-minute intervals)")
        print(f"Duration: {len(results_df) * 10 / 60:.1f} hours")
        print("\n" + "-"*70)
        print("POWER GENERATION SUMMARY")
        print("-"*70)
        print(f"Total Actual Power:     {results_df['Actual'].sum():15,.2f} kW")
        print(f"Total Predicted Power:  {results_df['Predicted'].sum():15,.2f} kW")
        print(f"Average Actual:         {results_df['Actual'].mean():15,.2f} kW")
        print(f"Average Predicted:      {results_df['Predicted'].mean():15,.2f} kW")
        print(f"Peak Actual:            {results_df['Actual'].max():15,.2f} kW")
        print(f"Peak Predicted:         {results_df['Predicted'].max():15,.2f} kW")
        print("\n" + "-"*70)
        print("DEVIATION ANALYSIS")
        print("-"*70)
        print(f"Mean Absolute Error:    {results_df['Absolute_Error'].mean():15,.2f} kW")
        print(f"Maximum Deviation:      {results_df['Absolute_Error'].max():15,.2f} kW")
        print(f"Mean % Error:           {results_df['Percentage_Error'].mean():15,.2f}%")
        print("\n" + "-"*70)
        print("HOURLY BREAKDOWN")
        print("-"*70)
        hourly = results_df.groupby('Hour').agg({
            'Actual': 'mean',
            'Predicted': 'mean',
            'Absolute_Error': 'mean'
        }).round(2)
        print(hourly.to_string())
        print("\n" + "-"*70)
        print("TURBINE-WISE PERFORMANCE")
        print("-"*70)
        turbine_summary = results_df.groupby('Turbine_ID').agg({
            'Actual': 'mean',
            'Predicted': 'mean',
            'Absolute_Error': 'mean'
        }).round(2)
        print(turbine_summary.to_string())
    def save_results(self, results_df, metrics):
        logger.info("Saving results to files...")
        results_path = os.path.join(self.output_dir, 'prediction_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved: {results_path}")
        self.execution_metadata['end_time'] = datetime.now().isoformat()
        self.execution_metadata['metrics'] = metrics

        metadata_path = os.path.join(self.output_dir, 'execution_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.execution_metadata, f, indent=4)
    def run_prediction(self, model_type='random_forest', specific_date=None, date_range=None, specific_timestamp=None):
        """
        Run prediction for specific date, date range, or exact timestamp
        Args:
            model_type: ML model type
            specific_date: 'YYYY-MM-DD' for single day
            date_range: ('YYYY-MM-DD', 'YYYY-MM-DD') for range
            specific_timestamp: ('DD-MM-YYYY', 'HH:MM') for exact timestamp
        """
        try:
            print("\n" + "="*70)
            print("WIND POWER PREDICTION SYSTEM")
            if specific_timestamp:
                print(f"Predicting for: {specific_timestamp[0]} at {specific_timestamp[1]}")
            elif specific_date:
                print(f"Predicting for: {specific_date}")
            elif date_range:
                print(f"Predicting for: {date_range[0]} to {date_range[1]}")
            print("="*70)
            print("\n[1/6] Loading data...")
            df = self.load_and_prepare_data()

            print("\n[2/6] Engineering features...")
            df = self.create_features(df)
            print("\n[3/6] Splitting data...")
            self.train_data, self.test_data = self.split_train_test(df)
            print(f"\n[4/6] Training {model_type} model...")
            self.train_model(model_type=model_type)
            # Handle specific timestamp prediction
            if specific_timestamp:
                print("\n[5/6] Generating prediction for specific timestamp...")
                results = self.predict_specific_timestamp(specific_timestamp[0], specific_timestamp[1])
                print("\n[6/6] Displaying results...")
                self.display_timestamp_prediction(results)
                print("\n" + "="*70)
                print("✓ PREDICTION COMPLETE")
                print("="*70)
                return results, None
            # Handle date/range predictions
            else:
                if specific_date or date_range:
                    print("\n[3b/6] Filtering test data...")
                    if specific_date:
                        target_date = pd.to_datetime(specific_date)
                        self.test_data = self.test_data[
                            self.test_data['Timestamp'].dt.date == target_date.date()
                        ].copy()
                    elif date_range:
                        start_date = pd.to_datetime(date_range[0])
                        end_date = pd.to_datetime(date_range[1])
                        self.test_data = self.test_data[
                            (self.test_data['Timestamp'].dt.date >= start_date.date()) &
                            (self.test_data['Timestamp'].dt.date <= end_date.date())
                        ].copy()
                print("\n[5/6] Generating predictions...")
                predictions, actual = self.predict()
                print("\n[6/6] Analyzing results...")
                results_df = self.calculate_errors(actual, predictions)
                metrics = self.calculate_metrics(actual, predictions)
                self.save_results(results_df, metrics)
                if specific_date and len(results_df) <= 144:  # Single day
                    self.generate_daily_summary(results_df)
                print("\n" + "="*70)
                print("✓ PREDICTION COMPLETE")
                print("="*70)
                return results_df, metrics
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise
# Main execution
if __name__ == "__main__":
    BASE_PATH = r"/content"
    TRAIN_FILES = [
        'Panapatty_.2018_scada_data.csv',
        'Panapatty_.2019_scada_data.csv',
        'Panapatty_.2020_scada_data.csv'
    ]
    TEST_FILE = 'Panapatty_.2021_scada_data.csv'
    OUTPUT_DIR = 'output'
    MODEL_TYPE = 'random_forest'
    if not os.path.isdir(BASE_PATH):
        print(f"\n❌ ERROR: Directory not found: {BASE_PATH}")
        sys.exit(1)
    # User input for prediction mode
    print("\n" + "="*70)
    print("WIND POWER PREDICTION SYSTEM")
    print("="*70)
    print("\nPrediction Mode:")
    print("  1. Full Year (2021)")
    print("  2. Specific Day")
    print("  3. Date Range")
    print("  4. Specific Date & Time")
    mode = input("\nSelect mode (1/2/3/4): ").strip()
    specific_date = None
    date_range = None
    specific_timestamp = None
    if mode == '2':
        date_input = input("Enter date (YYYY-MM-DD): ").strip()
        try:
            pd.to_datetime(date_input)
            specific_date = date_input
        except:
            print("❌ Invalid date format. Using full year.")
    elif mode == '3':
        start = input("Enter start date (YYYY-MM-DD): ").strip()
        end = input("Enter end date (YYYY-MM-DD): ").strip()
        try:
            pd.to_datetime(start)
            pd.to_datetime(end)
            date_range = (start, end)
        except:
            print("❌ Invalid date format. Using full year.")
    elif mode == '4':
        date_input = input("Enter date (DD-MM-YYYY): ").strip()
        time_input = input("Enter time (HH:MM): ").strip()
        try:
            # Validate inputs
            pd.to_datetime(f"{date_input} {time_input}", dayfirst=True)
            specific_timestamp = (date_input, time_input)
        except:
            print("❌ Invalid date/time format. Using full year.")
    # Initialize and run
    predictor = WindPowerPredictor(
        base_path=BASE_PATH,
        train_files=TRAIN_FILES,
        test_file=TEST_FILE,
        output_dir=OUTPUT_DIR
    )
    results, metrics = predictor.run_prediction(
        model_type=MODEL_TYPE,
        specific_date=specific_date,
        date_range=date_range,
        specific_timestamp=specific_timestamp
    )
    print(f"\n📁 Results saved to: {OUTPUT_DIR}/")
    print("="*70 + "\n")
def run_cloud_prediction(model_type='random_forest', date_range=None):
    """Bridge function for cloud environment deployment."""
    predictor = WindPowerPredictor(
        base_path="data", # Matches the DATA_DIR in App.py
        train_files=[
            'Panapatty_.2018_scada_data.csv',
            'Panapatty_.2019_scada_data.csv',
            'Panapatty_.2020_scada_data.csv'
        ],
        test_file='Panapatty_.2021_scada_data.csv',
        output_dir='output'
    )

    results_df, metrics = predictor.run_prediction(
        model_type=model_type,
        date_range=date_range
    )

    return results_df, metrics

if __name__ == "__main__":
    # This prevents the script from running interactive 'input()' prompts on Render
    if os.environ.get("RENDER") != "true":
        BASE_PATH = "data"
        TRAIN_FILES = [
            'Panapatty_.2018_scada_data.csv',
            'Panapatty_.2019_scada_data.csv',
            'Panapatty_.2020_scada_data.csv'
        ]
        TEST_FILE = 'Panapatty_.2021_scada_data.csv'
        
        predictor = WindPowerPredictor(base_path=BASE_PATH, train_files=TRAIN_FILES, test_file=TEST_FILE)
        # Add local test calls here if needed