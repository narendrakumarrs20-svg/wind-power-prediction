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
    - Date range prediction
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

    def split_train_test(self, df, date_range=None):
        """
        Split data - now supports date range filtering

        Args:
            df: Combined DataFrame
            date_range: Tuple of ('YYYY-MM-DD', 'YYYY-MM-DD') for range prediction
        """
        logger.info("Splitting data into train and test sets...")

        train_df = df[df['Source_File'].isin(self.train_files)].copy()
        test_df = df[df['Source_File'] == self.test_file].copy()

        # Filter test data by date range
        if date_range:
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
            raise ValueError(f"No data found for the specified date range")

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

    def run_prediction(self, model_type='random_forest', date_range=None):
        """
        Run prediction for full year or date range

        Args:
            model_type: ML model type
            date_range: ('YYYY-MM-DD', 'YYYY-MM-DD') for range
        """
        try:
            print("\n" + "="*70)
            print("WIND POWER PREDICTION SYSTEM")
            if date_range:
                print(f"Predicting for: {date_range[0]} to {date_range[1]}")
            else:
                print("Predicting for: Full Year 2021")
            print("="*70)

            print("\n[1/6] Loading data...")
            df = self.load_and_prepare_data()

            print("\n[2/6] Engineering features...")
            df = self.create_features(df)

            print("\n[3/6] Splitting data...")
            self.train_data, self.test_data = self.split_train_test(df, date_range=date_range)

            print(f"\n[4/6] Training {model_type} model...")
            self.train_model(model_type=model_type)

            print("\n[5/6] Generating predictions...")
            predictions, actual = self.predict()

            print("\n[6/6] Analyzing results...")
            results_df = self.calculate_errors(actual, predictions)
            metrics = self.calculate_metrics(actual, predictions)

            self.save_results(results_df, metrics)

            print("\n" + "="*70)
            print("✓ PREDICTION COMPLETE")
            print("="*70)

            return results_df, metrics

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise


def run_cloud_prediction(model_type='random_forest', date_range=None):
    """Bridge function for cloud environment deployment."""
    predictor = WindPowerPredictor(
        base_path="data",  # Matches the DATA_DIR in App.py
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


# Main execution
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
        print("  2. Date Range")

        mode = input("\nSelect mode (1/2): ").strip()

        date_range = None

        if mode == '2':
            start = input("Enter start date (YYYY-MM-DD): ").strip()
            end = input("Enter end date (YYYY-MM-DD): ").strip()
            try:
                pd.to_datetime(start)
                pd.to_datetime(end)
                date_range = (start, end)
            except:
                print("❌ Invalid date format. Using full year.")

        # Initialize and run
        predictor = WindPowerPredictor(
            base_path=BASE_PATH,
            train_files=TRAIN_FILES,
            test_file=TEST_FILE,
            output_dir=OUTPUT_DIRimport pandas as pd
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
import requests

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
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except Exception:
    RandomForestRegressor = None
    StandardScaler = None
    mean_absolute_error = None
    mean_squared_error = None
    r2_score = None


class WeatherAPIIntegration:
    """
    Integrates external weather APIs to enhance predictions
    
    Supported APIs:
    1. OpenWeatherMap (Free tier: 1000 calls/day)
    2. Visual Crossing Weather (Free tier: 1000 records/day)
    3. Open-Meteo (Free, no API key required)
    """
    
    def __init__(self, api_key=None, api_provider='open-meteo'):
        """
        Args:
            api_key: API key for paid services (OpenWeatherMap, Visual Crossing)
            api_provider: 'open-meteo', 'openweathermap', or 'visualcrossing'
        """
        self.api_key = api_key
        self.api_provider = api_provider
        self.cache = {}
        
    def fetch_weather_data(self, latitude, longitude, start_date, end_date):
        """
        Fetch historical weather data
        
        Returns:
            DataFrame with additional weather features
        """
        try:
            if self.api_provider == 'open-meteo':
                return self._fetch_open_meteo(latitude, longitude, start_date, end_date)
            elif self.api_provider == 'openweathermap':
                return self._fetch_openweathermap(latitude, longitude, start_date, end_date)
            elif self.api_provider == 'visualcrossing':
                return self._fetch_visualcrossing(latitude, longitude, start_date, end_date)
            else:
                logger.warning(f"Unknown API provider: {self.api_provider}")
                return None
        except Exception as e:
            logger.error(f"Weather API fetch failed: {e}")
            return None
    
    def _fetch_open_meteo(self, lat, lon, start_date, end_date):
        """
        Open-Meteo Historical Weather API (FREE, no key required)
        
        API: https://archive-api.open-meteo.com/v1/archive
        Documentation: https://open-meteo.com/en/docs/historical-weather-api
        
        Provides: wind speed/direction at multiple heights, temperature, pressure, humidity, cloud cover
        """
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        
        # Parameters with exact API variable names
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': ','.join([
                'temperature_2m',           # Temperature at 2 meters (°C)
                'relativehumidity_2m',      # Relative humidity at 2m (%)
                'dewpoint_2m',              # Dew point temperature (°C)
                'pressure_msl',             # Atmospheric pressure at sea level (hPa)
                'surface_pressure',         # Surface pressure (hPa)
                'cloudcover',               # Total cloud cover (%)
                'windspeed_10m',            # Wind speed at 10m (km/h)
                'windspeed_100m',           # Wind speed at 100m (km/h)
                'winddirection_10m',        # Wind direction at 10m (°)
                'winddirection_100m',       # Wind direction at 100m (°)
                'windgusts_10m',            # Wind gusts at 10m (km/h)
                'precipitation',            # Total precipitation (mm)
                'weathercode'               # Weather condition code
            ]),
            'timezone': 'Asia/Kolkata',     # Use IST timezone for India
            'windspeed_unit': 'ms',         # Return wind speed in m/s (matches your data)
            'temperature_unit': 'celsius'   # Celsius temperature
        }
        
        logger.info(f"🌐 Fetching Open-Meteo data for {start_date} to {end_date}...")
        logger.info(f"   Location: ({lat}, {lon})")
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if data was returned
            if 'hourly' not in data:
                logger.error(f"No hourly data in API response: {data}")
                return None
            
            hourly = data['hourly']
            
            # Create DataFrame with all available fields
            df = pd.DataFrame({
                'Timestamp': pd.to_datetime(hourly['time']),
                'API_Temperature': hourly.get('temperature_2m'),
                'API_Humidity': hourly.get('relativehumidity_2m'),
                'API_DewPoint': hourly.get('dewpoint_2m'),
                'API_Pressure_MSL': hourly.get('pressure_msl'),
                'API_Pressure_Surface': hourly.get('surface_pressure'),
                'API_CloudCover': hourly.get('cloudcover'),
                'API_WindSpeed_10m': hourly.get('windspeed_10m'),
                'API_WindSpeed_100m': hourly.get('windspeed_100m'),
                'API_WindDir_10m': hourly.get('winddirection_10m'),
                'API_WindDir_100m': hourly.get('winddirection_100m'),
                'API_WindGusts_10m': hourly.get('windgusts_10m'),
                'API_Precipitation': hourly.get('precipitation'),
                'API_WeatherCode': hourly.get('weathercode')
            })
            
            # Remove columns that are all None
            df = df.dropna(axis=1, how='all')
            
            logger.info(f"✓ Successfully fetched {len(df)} weather records from Open-Meteo")
            logger.info(f"✓ Weather features: {list(df.columns[1:])}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Open-Meteo API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error processing Open-Meteo data: {e}")
            return None
    
    def _fetch_visualcrossing(self, lat, lon, start_date, end_date):
        """
        Visual Crossing Weather API (Free tier: 1000 records/day)
        """
        if not self.api_key:
            logger.warning("Visual Crossing requires API key")
            return None
            
        base_url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{start_date}/{end_date}"
        
        params = {
            'key': self.api_key,
            'include': 'hours',
            'elements': 'datetime,temp,humidity,pressure,windspeed,winddir,cloudcover,precip'
        }
        
        logger.info(f"Fetching Visual Crossing data...")
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        records = []
        for day in data['days']:
            for hour in day.get('hours', []):
                records.append({
                    'Timestamp': pd.to_datetime(f"{day['datetime']} {hour['datetime']}"),
                    'API_Temperature': hour.get('temp'),
                    'API_Humidity': hour.get('humidity'),
                    'API_Pressure': hour.get('pressure'),
                    'API_WindSpeed': hour.get('windspeed'),
                    'API_WindDir': hour.get('winddir'),
                    'API_CloudCover': hour.get('cloudcover'),
                    'API_Precipitation': hour.get('precip')
                })
        
        df = pd.DataFrame(records)
        logger.info(f"Successfully fetched {len(df)} weather records from Visual Crossing")
        return df
    
    def _fetch_openweathermap(self, lat, lon, start_date, end_date):
        """
        OpenWeatherMap Historical API (Requires paid subscription)
        """
        if not self.api_key:
            logger.warning("OpenWeatherMap requires API key")
            return None
            
        logger.warning("OpenWeatherMap historical data requires paid subscription")
        return None


class WindPowerPredictor:
    """
    Enhanced Cloud-Based Smart Scheduling System for Renewable Power Generation
    
    Now with:
    - Random Forest as default (single mode)
    - Weather API integration for accuracy boost
    - Advanced feature engineering
    """

    COLUMN_MAP = {
        'Wind Speed at 78.5 mtr': 'Wind Speed (m/s)',
        'Wind Direction at 78.5 mtr': 'Wind Direction (degrees)',
        'Ambient Temp at 78.5 mtr': 'Ambient Temperature (°C)',
        'Active_Power 78.5 mtr': 'Power Generated (kW)',
    }
    REQUIRED_COLUMNS = list(COLUMN_MAP.values()) + ['Rotor RPM']

    def __init__(self, base_path, train_files, test_file, output_dir='output',
                 weather_api_key=None, weather_api_provider='open-meteo',
                 turbine_location=(10.9167, 78.1333)):  # Default: Panapatty, Tamil Nadu
        """
        Args:
            base_path: Data directory
            train_files: List of training CSV files
            test_file: Test CSV file
            output_dir: Output directory
            weather_api_key: API key for weather services (optional for open-meteo)
            weather_api_provider: 'open-meteo', 'openweathermap', or 'visualcrossing'
            turbine_location: (latitude, longitude) tuple
        """
        self.base_path = base_path
        self.train_files = train_files
        self.test_file = test_file
        self.output_dir = output_dir
        self.turbine_location = turbine_location
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.train_data = None
        self.test_data = None
        self.use_weather_api = False
        
        # Weather API integration
        self.weather_api = WeatherAPIIntegration(
            api_key=weather_api_key,
            api_provider=weather_api_provider
        )
        
        self.execution_metadata = {
            'start_time': datetime.now().isoformat(),
            'model_type': 'Random Forest',  # Fixed default
            'train_samples': 0,
            'test_samples': 0,
            'weather_api_used': False,
            'weather_api_provider': weather_api_provider
        }

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initialized WindPowerPredictor with output directory: {output_dir}")

        try:
            self._ensure_ml_packages()
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            globals().update({
                'RandomForestRegressor': RandomForestRegressor,
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
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy', 'scipy', 'scikit-learn', 'requests'])
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

    def load_and_prepare_data(self, use_weather_api=True):
        """
        Load SCADA data and optionally integrate weather API data
        
        Args:
            use_weather_api: Whether to fetch and merge weather API data
        """
        logger.info("Starting data loading and preparation...")
        self.use_weather_api = use_weather_api
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
        
        # Integrate Weather API data if enabled
        if use_weather_api:
            df = self._integrate_weather_data(df)
        
        df = df.set_index('Timestamp')
        df = df.asfreq('10T', method='ffill')
        df = df.reset_index()

        logger.info(f"Total records loaded: {len(df)}")
        return df

    def _integrate_weather_data(self, df):
        """
        Merge Open-Meteo weather API data with SCADA data
        
        This dramatically improves prediction accuracy by adding:
        - Multi-height wind measurements
        - Atmospheric conditions
        - Weather patterns
        """
        try:
            start_date = df['Timestamp'].min().strftime('%Y-%m-%d')
            end_date = df['Timestamp'].max().strftime('%Y-%m-%d')
            
            lat, lon = self.turbine_location
            
            logger.info(f"🌐 Requesting weather data from Open-Meteo...")
            logger.info(f"   Date range: {start_date} to {end_date}")
            logger.info(f"   Location: {lat}°N, {lon}°E (Panapatty Wind Farm)")
            
            weather_df = self.weather_api.fetch_weather_data(
                latitude=lat,
                longitude=lon,
                start_date=start_date,
                end_date=end_date
            )
            
            if weather_df is not None and len(weather_df) > 0:
                logger.info(f"✓ Received {len(weather_df)} hourly weather records")
                
                # Round timestamps to nearest 10 minutes for matching SCADA data
                df['Timestamp_Round'] = df['Timestamp'].dt.round('H')  # Round to hour
                weather_df['Timestamp_Round'] = weather_df['Timestamp'].dt.round('H')
                
                # Merge on rounded timestamp
                df_before = len(df)
                df = df.merge(
                    weather_df, 
                    on='Timestamp_Round', 
                    how='left', 
                    suffixes=('', '_weather_api')
                )
                
                # Clean up duplicate timestamp columns
                df = df.drop(columns=['Timestamp_Round'], errors='ignore')
                if 'Timestamp_weather_api' in df.columns:
                    df = df.drop(columns=['Timestamp_weather_api'])
                
                # Get API column names
                api_cols = [col for col in df.columns if col.startswith('API_')]
                
                if len(api_cols) > 0:
                    # Fill missing API values with interpolation
                    for col in api_cols:
                        missing_before = df[col].isna().sum()
                        # Forward fill, then backward fill, then interpolate
                        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                        df[col] = df[col].interpolate(method='linear', limit_direction='both')
                        missing_after = df[col].isna().sum()
                        
                        if missing_before > 0:
                            logger.info(f"   {col}: filled {missing_before - missing_after} missing values")
                    
                    logger.info(f"✓ Weather API integration successful!")
                    logger.info(f"✓ Added {len(api_cols)} weather features:")
                    for col in sorted(api_cols):
                        logger.info(f"   - {col}")
                    
                    self.execution_metadata['weather_api_used'] = True
                    self.execution_metadata['weather_features_count'] = len(api_cols)
                else:
                    logger.warning("⚠ No API columns found after merge")
            else:
                logger.warning("⚠ Weather API fetch returned no data - continuing without API enhancement")
                
        except Exception as e:
            logger.warning(f"⚠ Weather API integration failed: {e}")
            logger.warning("   Continuing with SCADA data only...")
            import traceback
            logger.debug(traceback.format_exc())
            
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

        nat_count = int(dt.isna().sum())
        if nat_count > 0:
            logger.warning(f"{nat_count} timestamps could not be parsed.")

        return dt

    def create_features(self, df):
        """Enhanced feature engineering with weather API data"""
        logger.info("Creating engineered features...")
        df = df.copy()

        # Wind direction encoding
        if 'Wind Direction (degrees)' in df.columns:
            df['wind_x'] = np.sin(np.radians(df['Wind Direction (degrees)']))
            df['wind_y'] = np.cos(np.radians(df['Wind Direction (degrees)']))
        else:
            df['wind_x'] = 0
            df['wind_y'] = 0

        # Temporal features
        df['hour'] = df['Timestamp'].dt.hour
        df['day_of_year'] = df['Timestamp'].dt.dayofyear
        df['month'] = df['Timestamp'].dt.month
        df['season'] = (df['month'] % 12 + 3) // 3
        df['day_of_week'] = df['Timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Lag features
        df['wind_speed_lag1'] = df['Wind Speed (m/s)'].shift(1)
        df['wind_speed_lag6'] = df['Wind Speed (m/s)'].shift(6)
        df['power_lag1'] = df['Power Generated (kW)'].shift(1)
        df['power_lag6'] = df['Power Generated (kW)'].shift(6)

        # Rolling window features
        df['wind_speed_roll6'] = df['Wind Speed (m/s)'].rolling(window=6, min_periods=1).mean()
        df['wind_speed_roll12'] = df['Wind Speed (m/s)'].rolling(window=12, min_periods=1).mean()
        df['wind_speed_roll24'] = df['Wind Speed (m/s)'].rolling(window=24, min_periods=1).mean()
        df['power_roll6'] = df['Power Generated (kW)'].rolling(window=6, min_periods=1).mean()
        df['power_roll12'] = df['Power Generated (kW)'].rolling(window=12, min_periods=1).mean()
        df['wind_speed_std6'] = df['Wind Speed (m/s)'].rolling(window=6, min_periods=1).std()
        df['temp_roll6'] = df['Ambient Temperature (°C)'].rolling(window=6, min_periods=1).mean()

        # Physical model features
        df['wind_temp_interaction'] = df['Wind Speed (m/s)'] * df['Ambient Temperature (°C)']
        df['wind_speed_squared'] = df['Wind Speed (m/s)'] ** 2
        df['wind_speed_cubed'] = df['Wind Speed (m/s)'] ** 3
        
        # Air density calculation (affects power output)
        df['air_density'] = 1.225 * (288.15 / (df['Ambient Temperature (°C)'] + 273.15))
        
        # Theoretical power based on wind speed
        df['theoretical_power'] = np.where(
            df['Wind Speed (m/s)'] < 3, 0,  # Cut-in speed
            np.where(
                df['Wind Speed (m/s)'] > 25, 0,  # Cut-out speed
                0.5 * df['air_density'] * np.pi * (39**2) * (df['Wind Speed (m/s)']**3) * 0.4
            )
        )
        
        # Wind speed change (turbulence indicator)
        df['wind_speed_change'] = df['Wind Speed (m/s)'].diff()
        df['wind_turbulence'] = df['wind_speed_std6'] / (df['wind_speed_roll6'] + 0.1)

        # Weather API features integration (Open-Meteo provides rich atmospheric data)
        if self.use_weather_api:
            # Pressure features
            if 'API_Pressure_MSL' in df.columns:
                df['pressure_change'] = df['API_Pressure_MSL'].diff()
                df['pressure_roll6'] = df['API_Pressure_MSL'].rolling(window=6, min_periods=1).mean()
            
            # Humidity interaction with temperature
            if 'API_Humidity' in df.columns:
                df['humidity_temp_interaction'] = df['API_Humidity'] * df['Ambient Temperature (°C)']
                df['dewpoint_spread'] = df['Ambient Temperature (°C)'] - df.get('API_DewPoint', 0)
            
            # Cloud cover impact on wind patterns
            if 'API_CloudCover' in df.columns:
                df['cloud_wind_interaction'] = df['API_CloudCover'] * df['Wind Speed (m/s)']
            
            # Multi-height wind analysis (CRITICAL for turbine at 78.5m)
            if 'API_WindSpeed_10m' in df.columns and 'API_WindSpeed_100m' in df.columns:
                # Wind shear coefficient (how wind speed changes with height)
                df['wind_shear'] = (df['API_WindSpeed_100m'] - df['API_WindSpeed_10m']) / 90
                df['wind_shear_ratio'] = df['API_WindSpeed_100m'] / (df['API_WindSpeed_10m'] + 0.1)
                
                # Interpolate wind speed at turbine height (78.5m)
                # Using logarithmic wind profile interpolation
                df['API_WindSpeed_78m'] = df['API_WindSpeed_10m'] + (
                    (df['API_WindSpeed_100m'] - df['API_WindSpeed_10m']) * 
                    (np.log(78.5) - np.log(10)) / (np.log(100) - np.log(10))
                )
                
                # Wind direction consistency between heights
                if 'API_WindDir_10m' in df.columns and 'API_WindDir_100m' in df.columns:
                    # Calculate directional difference (accounting for 0/360 wrap)
                    dir_diff = np.abs(df['API_WindDir_100m'] - df['API_WindDir_10m'])
                    dir_diff = np.minimum(dir_diff, 360 - dir_diff)
                    df['wind_direction_consistency'] = 1 - (dir_diff / 180)  # 1 = same direction
            
            # Wind gusts impact
            if 'API_WindGusts_10m' in df.columns and 'API_WindSpeed_10m' in df.columns:
                df['gust_factor'] = df['API_WindGusts_10m'] / (df['API_WindSpeed_10m'] + 0.1)
                df['wind_variability'] = df['API_WindGusts_10m'] - df['API_WindSpeed_10m']
            
            # Precipitation effect on turbine efficiency
            if 'API_Precipitation' in df.columns:
                df['is_raining'] = (df['API_Precipitation'] > 0).astype(int)
                df['rain_wind_interaction'] = df['API_Precipitation'] * df['Wind Speed (m/s)']
            
            # Weather condition encoding
            if 'API_WeatherCode' in df.columns:
                # WMO weather codes: 0=clear, 1-3=clouds, 45-48=fog, 51-99=precipitation
                df['weather_clear'] = (df['API_WeatherCode'] == 0).astype(int)
                df['weather_rainy'] = (df['API_WeatherCode'] >= 51).astype(int)
            
            logger.info("✓ Advanced weather API features engineered")

        df = df.dropna()
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df

    def split_train_test(self, df, date_range=None):
        """Split data with optional date range filtering"""
        logger.info("Splitting data into train and test sets...")

        train_df = df[df['Source_File'].isin(self.train_files)].copy()
        test_df = df[df['Source_File'] == self.test_file].copy()

        if date_range:
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
            raise ValueError(f"No data found for the specified date range")

        return train_df, test_df

    def prepare_features(self, df):
        """Prepare feature matrix with all engineered features"""
        base_features = [
            'Wind Speed (m/s)', 'wind_x', 'wind_y',
            'Ambient Temperature (°C)', 'Rotor RPM',
            'hour', 'day_of_year', 'season', 'day_of_week', 'is_weekend',
            'wind_speed_lag1', 'wind_speed_lag6',
            'power_lag1', 'power_lag6',
            'wind_speed_roll6', 'wind_speed_roll12', 'wind_speed_roll24',
            'power_roll6', 'power_roll12',
            'wind_speed_std6', 'temp_roll6',
            'wind_temp_interaction', 'wind_speed_squared', 'wind_speed_cubed',
            'air_density', 'theoretical_power', 'wind_speed_change', 'wind_turbulence'
        ]
        
        # Add weather API features if available
        api_features = [col for col in df.columns if col.startswith('API_') or 
                       col in ['pressure_change', 'humidity_temp_interaction', 
                              'cloud_wind_interaction', 'wind_shear', 'wind_shear_ratio']]
        
        feature_cols = base_features + api_features + ['Turbine_ID']
        
        # Filter only available columns
        available_features = [col for col in feature_cols if col in df.columns]

        df = pd.get_dummies(df, columns=['Turbine_ID'], prefix='TID', drop_first=True)

        feature_cols_final = [col for col in available_features if col != 'Turbine_ID']
        feature_cols_final += [col for col in df.columns if col.startswith('TID_')]

        X = df[feature_cols_final].copy()
        y = df['Power Generated (kW)'].copy()

        return X, y, feature_cols_final

    def train_model(self):
        """Train Random Forest model with optimized hyperparameters"""
        logger.info("Training Random Forest model...")

        X_train, y_train, self.feature_columns = self.prepare_features(self.train_data)
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Optimized Random Forest configuration
        self.model = RandomForestRegressor(
            n_estimators=200,        # Increased from 100
            max_depth=25,            # Increased from 20
            min_samples_split=5,     # Decreased from 10 for more splits
            min_samples_leaf=2,      # Decreased from 5 for finer granularity
            max_features='sqrt',     # Feature subsampling
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        self.model.fit(X_train_scaled, y_train)
        train_score = self.model.score(X_train_scaled, y_train)

        logger.info(f"Training complete. R² score: {train_score:.4f}")
        self.execution_metadata['train_r2'] = train_score

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 5 features: {feature_importance.head(5)['feature'].tolist()}")

        return self.model

    def predict(self):
        """Make predictions on test data"""
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

    def calculate_errors(self, actual, predicted):
        """Calculate detailed error metrics"""
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
        """Calculate performance metrics"""
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

    def save_results(self, results_df, metrics):
        """Save results and metadata"""
        logger.info("Saving results to files...")

        results_path = os.path.join(self.output_dir, 'prediction_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved: {results_path}")

        self.execution_metadata['end_time'] = datetime.now().isoformat()
        self.execution_metadata['metrics'] = metrics

        metadata
        )

        results, metrics = predictor.run_prediction(
            model_type=MODEL_TYPE,
            date_range=date_range
        )

        print(f"\n📁 Results saved to: {OUTPUT_DIR}/")
        print("="*70 + "\n")