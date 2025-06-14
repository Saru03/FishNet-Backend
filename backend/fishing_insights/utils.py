import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
from netCDF4 import Dataset
import xarray as xr
import xgboost as xgb
import joblib
from django.conf import settings
from dateutil import parser
import cartopy
import earthaccess
from huggingface_hub import hf_hub_download


print("DEBUG:: ENV USERNAME:", os.environ.get("EARTHDATA_USERNAME"))
print("DEBUG:: ENV PASSWORD:", os.environ.get("EARTHDATA_PASSWORD"))

CACHE_DIR = os.path.join(settings.BASE_DIR, ".hf-cache")
# Authenticate earthaccess
auth = earthaccess.login(strategy="environment")
print("Earthaccess login successful.")

# Load models and other files
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml-models', 'pfz_model.json')
SCALER_PATH = os.path.join(settings.BASE_DIR, 'ml-models', 'pfz_scaler.pkl')
RF_IMPUTER_PATH = hf_hub_download(
    repo_id="saru03/Fishnet-Imputer",
    filename="rf_imputer_model.pkl",
    cache_dir=CACHE_DIR
)
FEATURES_PATH = os.path.join(settings.BASE_DIR, 'ml-models', 'pfz_features.txt')

# Load model
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
print("Model loaded.")

# Load scaler and imputer
scaler = joblib.load(SCALER_PATH)
rf_imputer = joblib.load(RF_IMPUTER_PATH)
print("Scaler and imputer loaded.")

# Load feature names
with open(FEATURES_PATH, 'r') as f:
    feature_names = [line.strip() for line in f]
print("Features loaded:", feature_names)


def build_feature_vector_for_imputer(sst, sst_grad, chlor, weather, effort_hours):
    # Initialize all features with default value 0
    feature_dict = {f: 0 for f in feature_names}
    
    # Assign known features
    for f in feature_names:
        if f == 'sst':
            feature_dict[f] = sst
        elif f == 'sst_gradient':
            feature_dict[f] = sst_grad
        elif f == 'chlorophyll':
            feature_dict[f] = chlor
        elif f == 'temperature':
            feature_dict[f] = weather.get('temp', 0)
        elif f == 'humidity':
            feature_dict[f] = weather.get('humidity', 0)
        elif f == 'pressure':
            feature_dict[f] = weather.get('pressure', 0)
        elif f == 'precipitation':
            feature_dict[f] = weather.get('precipitation', 0)
        elif f == 'mean_sea_level_pressure':
            feature_dict[f] = weather.get('pressure', 0)
        elif f.startswith('front_'):
            # default value
            feature_dict[f] = 0.5
        elif f.startswith('persistence_'):
            # default value
            feature_dict[f] = 0.5
        elif f == 'fishing_effort_hours':
            feature_dict[f] = effort_hours
        # Add more features as needed
    return feature_dict

# CPUE dictionary
CPUE = {
    'drifting_longlines': 0.5,
    'purse_seine': 2.0,
    'pole_and_line': 1.0,
    'set_longlines': 0.8,
    'squid_jigging': 0.3,
    'trawlers': 1.5
}
print("CPUE dictionary initialized.")

class ProcessedData:
    @classmethod
    def fetch_data(cls, date_str, latitude, longitude):
        print(f"Fetching data for date: {date_str}, lat: {latitude}, lon: {longitude}")
        try:
            lat = float(latitude)
            lon = float(longitude)
            print(f"Parsed lat: {lat}, lon: {lon}")
        except Exception as e:
            print(f"Error parsing latitude/longitude: {e}")
            return {"error": "Invalid latitude or longitude"}

        formatted_date = datetime.strptime(date_str, '%Y-%m-%d')
        date_str_formatted = formatted_date.strftime('%Y%m%d')
        # sst_path = os.path.join(settings.BASE_DIR, f"AQUA_MODIS.{date_str_formatted}.L3m.DAY.SST.sst.4km.NRT.nc")
        # print(f"SST path: {sst_path}")

        # Fetch chlorophyll
        chl_value = cls._fetch_chlorophyll(lat, lon, date_str)
        print(f"Chlorophyll value: {chl_value}")

        # Fetch SST and gradient
        sst, sst_grad = cls._fetch_sst(lat, lon, date_str)
        print(f"SST: {sst}, SST Gradient: {sst_grad}")

        # Fetch weather
        weather = cls._fetch_weather(lat, lon, date_str)
        print(f"Weather data: {weather}")

        # Fetch fishing effort
        effort_hours, gear_types = cls._fetch_fishing_effort(lat, lon, date_str)
        print(f"Effort hours: {effort_hours}, gear types: {gear_types}")
        gear_type = gear_types[0] if gear_types else None
        print(f"Selected gear type: {gear_type}")

        # Calculate estimated catch
        cpue = CPUE.get(gear_type, 0)
        estimated_catch = effort_hours * cpue
        print(f"Estimated catch: {estimated_catch}")

        if estimated_catch <= 0:
            print("Estimated catch is non-positive, invoking imputer.")
            # Build the feature vector with all features
            feature_dict = build_feature_vector_for_imputer(sst, sst_grad, chl_value, weather, effort_hours)
            
            # Create DataFrame in the exact feature order
            feature_df = pd.DataFrame([feature_dict], columns=feature_names)
            
            # Fill any NaNs just in case
            feature_df.fillna(0, inplace=True)
            
            # Scale features
            X_input = scaler.transform(feature_df)
            # Ensure only the features used during training are passed
            imputer_features = [
                'sst', 'chlorophyll', 'sst_gradient',
                'front_0.3', 'front_0.4', 'front_0.5',
                'persistence_0.3', 'persistence_0.4', 'persistence_0.5',
                'temp', 'humidity', 'precipitation', 'mean_sea_level_pressure'
            ]

            # Select only these features in the correct order
            X_impute = feature_df[imputer_features]

            # Predict using the imputer
            predicted_catch = rf_imputer.predict(X_impute)[0]
            print(f"Imputed predicted catch: {predicted_catch}")
            
            # Use the imputed value
            estimated_catch = predicted_catch

        # Optional: print the features used for debugging
        print("Features for imputer:", feature_df.to_dict(orient='records')[0])

        # Prepare data for prediction
        data = {
            "date": date_str,
            "latitude": lat,
            "longitude": lon,
            "sst": sst,
            "sst_gradient": sst_grad,
            "chlorophyll": chl_value,
            "temperature": weather.get('temp'),
            "humidity": weather.get('humidity'),
            "pressure": weather.get('pressure'),
            "precipitation": weather.get('precipitation'),
            "fishing_effort_hours": effort_hours,
        }
        print("Data to predict:", data)
        result = cls._predict(data)
        print("Prediction result:", result)
        return result

    @staticmethod
    def _fetch_weather(lat, lon, date_str):
        print(f"Fetching weather for {lat}, {lon} on {date_str}")
        url = "https://api.open-meteo.com/v1/forecast"
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        response = requests.get(url, params={
            "latitude": lat,
            "longitude": lon,
            "daily": ["pressure_msl_mean", "relative_humidity_2m_mean", "temperature_2m_mean", "precipitation_sum"],
            "timezone": "auto",
            "past_days": 7,
	        "forecast_days": 1
        })
        print(f"Weather API response status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Weather API data received.")
            daily = data.get('daily', {})
            dates = pd.to_datetime(daily.get('time', []))
            print(f"Available dates in response: {dates}")
            idx = next((i for i, d in enumerate(dates) if d.date() == dt.date()), None)
            if idx is not None:
                print(f"Matching date index: {idx}")
                return {
                    'pressure': daily['pressure_msl_mean'][idx],
                    'humidity': daily['relative_humidity_2m_mean'][idx],
                    'temp': daily['temperature_2m_mean'][idx],
                    'precipitation': daily['precipitation_sum'][idx]
                }
            else:
                print("No matching date in weather data.")
        else:
            print(f"Weather fetch failed with status {response.status_code}")
        return {'pressure': np.nan, 'humidity': np.nan, 'temp': np.nan, 'precipitation': np.nan}

    @staticmethod
    def _fetch_sst(lat, lon, date_str):
        print(f"Fetching SST for {date_str} at lat: {lat}, lon: {lon}")
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        start_date = (date_obj - timedelta(days=1)).strftime("%Y-%m-%d")
        end_date = date_str

        sst_value = np.nan
        sst_gradient = np.nan
        try:
            results = earthaccess.search_data(
                short_name='MODISA_L3m_SST_NRT',
                granule_name="*.DAY*.4km*",
                temporal=(start_date, end_date),
            )
            print(f"Number of SST results found: {len(results)}")
            if not results:
                print("No SST data found for the specified date range.")
                return np.nan, np.nan

            granule = results[0]
            print(f"Using granule: {granule}")

            if not granule.cloud_hosted:
                print("Downloading SST data file.")
                paths = earthaccess.download([granule], "sst.nc")
                ds = xr.open_dataset(paths[0])
            else:
                print("Opening cloud-hosted SST dataset.")
                ds = xr.open_dataset(granule.online_access_url)

            print("Variables in SST dataset:", list(ds.data_vars))
            lats = ds['lat'].values
            lons = ds['lon'].values
            sst_var = ds['sst']

            print(f"SST data shape: {sst_var.shape}")
            # Handle multiple time steps if necessary
            sst_data = sst_var[0] if sst_var.ndim == 3 else sst_var[:]
            print(f"SST data shape after indexing: {sst_data.shape}")

            lat_idx = np.argmin(np.abs(lats - lat))
            lon_idx = np.argmin(np.abs(lons - lon))
            print(f"Nearest lat index: {lat_idx}, longitude index: {lon_idx}")

            sst_value = sst_data[lat_idx, lon_idx]
            sst_value=float(sst_value)
            print(f"SST value at location: {sst_value}")

            # Compute gradient
            grad_lat, grad_lon = np.gradient(sst_data)
            sst_gradient_array = np.sqrt(grad_lat**2 + grad_lon**2)
            sst_gradient = sst_gradient_array[lat_idx, lon_idx]
            print(f"SST gradient at location: {sst_gradient}")

            ds.close()
        except Exception as e:
            print(f"Error in _fetch_sst: {e}")

        return sst_value, sst_gradient

    @staticmethod
    def _fetch_chlorophyll(lat, lon, date_str):
        print(f"Fetching chlorophyll for {date_str} at lat: {lat}, lon: {lon}")
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        start_date = (date_obj - timedelta(days=3)).strftime("%Y-%m-%d")
        end_date = date_str

        chlor_value = np.nan  # Initialize outside try block to access in exception handling

        try:
            results = earthaccess.search_data(
                short_name='MODISA_L3m_CHL_NRT',
                granule_name="*.DAY*.4km*",
                temporal=(start_date, end_date),
            )
            print(f"Number of chlorophyll results: {len(results)}")
            if not results:
                print("No chlorophyll data found for the specified date range.")
                return np.nan

            granule = results[0]
            print(f"Using granule: {granule}")

            # Download dataset
            paths = earthaccess.download([granule], "chlorophyll.nc")
            ds = xr.open_dataset(paths[0])
            print("Variables in chlorophyll dataset:", list(ds.data_vars))
            chl_var_name = [v for v in ds.data_vars if 'chlor' in v.lower()]
            print(f"Detected chlorophyll variable(s): {chl_var_name}")

            if not chl_var_name:
                print("No chlorophyll variable found in dataset.")
                ds.close()
                return np.nan

            chl_var_name = chl_var_name[0]
            chl_var = ds[chl_var_name]
            print(f"Chlorophyll variable shape: {chl_var.shape}")

            # Find nearest grid point
            lat_idx = np.argmin(np.abs(chl_var['lat'].values - lat))
            lon_idx = np.argmin(np.abs(chl_var['lon'].values - lon))
            print(f"Nearest lat index: {lat_idx}, lon index: {lon_idx}")

            def get_value_at(i, j):
                try:
                    val = chl_var.values[i, j]
                    if np.isnan(val):
                        return None
                    return val
                except IndexError:
                    return None

            # Get initial value
            chlor_value = get_value_at(lat_idx, lon_idx)

            # If nan, check nearby grid points within a window (e.g., 3x3)
            if chlor_value is None:
                window_size = 3  # search one grid point around
                for di in range(-window_size, window_size + 1):
                    for dj in range(-window_size, window_size + 1):
                        i = lat_idx + di
                        j = lon_idx + dj
                        val = get_value_at(i, j)
                        if val is not None:
                            chlor_value = val
                            print(f"Found nearby valid chlorophyll value at ({i}, {j}): {val}")
                            ds.close()
                            return chlor_value
                print("No valid chlorophyll data in nearby grid points.")
            else:
                print(f"Chlorophyll value at exact location: {chlor_value}")

            ds.close()
            return chlor_value

        except Exception as e:
            print(f"Error in _fetch_chlorophyll: {e}")
            return np.nan

    @staticmethod
    def _fetch_fishing_effort(lat, lon, date_str):
        delta = 0.5
        geojson_geom = {
            "type": "Polygon",
            "coordinates": [[
                [lon - delta, lat - delta],
                [lon + delta, lat - delta],
                [lon + delta, lat + delta],
                [lon - delta, lat + delta],
                [lon - delta, lat - delta]
            ]]
        }
        url = 'https://gateway.api.globalfishingwatch.org/v2/4wings/stats/'
        # Add your API token here
        token='YOUR_TOKEN_HERE'
        headers = {'Authorization': f'Bearer {token}'}
        params = {
            'datasets[0]': 'public-global-fishing-effort:latest',
            'fields': 'activityHours,geartype',
            'date-range': f'{date_str},{date_str}',
            'geojson': json.dumps(geojson_geom)
        }
        print(f"Requesting fishing effort with params: {params}")
        try:
            resp = requests.get(url, params=params, headers=headers)
            print(f"Fishing effort response status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print("Fishing effort data received.")
                total_hours = sum(item.get('activityHours', 0) for item in data)
                gear_types = list(set(item.get('geartype') for item in data))
                print(f"Total activity hours: {total_hours}, gear types: {gear_types}")
                return total_hours, gear_types
            else:
                print(f"Failed to fetch fishing effort: {resp.status_code}")
        except Exception as e:
            print(f"Error fetching fishing effort: {e}")
        return 0, []
  
    @staticmethod
    def _predict(data):
        print("Starting prediction with data:", data)
        print("Starting prediction with data:", data)
        df = pd.DataFrame([data])
        df = pd.DataFrame([data])
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        print("Dataframe after date processing:", df)

        # Extract scalar values for features fetched from data
        sst = data['sst']
        sst_gradient = data['sst_gradient']
        chlorophyll = data['chlorophyll']
        temperature = data['temperature']
        humidity = data['humidity']
        pressure = data['pressure']
        precipitation = data['precipitation']
        fishing_hours = data.get('fishing_effort_hours', 0)
        estimated_catch = data.get('estimated_catch_tons', 0)  # if available

        # Convert to scalar floats if they are xr.DataArray or np.ndarray
        def to_scalar(val):
            if hasattr(val, 'item'):
                return val.item()
            elif isinstance(val, (np.ndarray,)):
                return float(val)
            elif val is None:
                return 0.0
            else:
                return float(val)

        sst = to_scalar(sst)
        sst_gradient = to_scalar(sst_gradient)
        chlorophyll = to_scalar(chlorophyll)
        temperature = to_scalar(temperature)
        humidity = to_scalar(humidity)
        pressure = to_scalar(pressure)
        precipitation = to_scalar(precipitation)
        fishing_hours = float(fishing_hours)
        estimated_catch = float(estimated_catch)

        # Assign these scalar values to df for later use
        df['lat'] = data.get('latitude', 0)
        df['lon'] = data.get('longitude', 0)
        df['temp'] = temperature
        df['mean_sea_level_pressure'] = pressure
        df['fishing_hours'] = fishing_hours
        df['estimated_catch_tons'] = estimated_catch

        # Also add any other features your model expects
        # For example, if model uses 'persistence_gradient_ratio'
        # you need to compute and add it here, e.g.,
        # df['persistence_gradient_ratio'] = some_value

        # Generate interaction features
        df['sst_chlorophyll'] = sst * chlorophyll
        df['sst_gradient_chlorophyll'] = sst_gradient * chlorophyll

        # Normalize ocean variables
        def norm(val, min_val, max_val):
            return (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5

        # For each feature to normalize, get min and max
        # (Make sure these are computed from your training data or set as constants)
        sst_min, sst_max = df['sst'].min(), df['sst'].max()
        chlor_min, chlor_max = df['chlorophyll'].min(), df['chlorophyll'].max()
        gradient_min, gradient_max = df['sst_gradient'].min(), df['sst_gradient'].max()

        df['sst_score'] = norm(sst, sst_min, sst_max)
        df['chlorophyll_score'] = norm(chlorophyll, chlor_min, chlor_max)
        df['gradient_score'] = norm(sst_gradient, gradient_min, gradient_max)

        # Compute ocean score
        df['ocean_score'] = 0.3 * df['sst_score'] + 0.4 * df['chlorophyll_score'] + 0.3 * df['gradient_score']

        # Add default values for missing features
        for col in ['front_0.3', 'front_0.4', 'front_0.5', 'persistence_0.3', 'persistence_0.4', 'persistence_0.5']:
            if col not in df.columns:
                df[col] = 0.5

        # Compute front and persistence scores
        df['front_score'] = (df['front_0.3'] + df['front_0.4'] + df['front_0.5']) / 3
        max_persistence = df[['persistence_0.3', 'persistence_0.4', 'persistence_0.5']].max(axis=1).iloc[0]
        df['persistence_score'] = 1 - ((df['persistence_0.3'] + df['persistence_0.4'] + df['persistence_0.5']) / (3 * max_persistence))

        df['persistence_gradient_ratio'] = df['persistence_score'] / (df['sst_gradient'] + 1e-6)
        # Weather scores
        temp_min, temp_max = df['temperature'].min(), df['temperature'].max()
        precip_max = df['precipitation'].max() if df['precipitation'].max() > 0 else 1
        pressure_min, pressure_max = df['pressure'].min(), df['pressure'].max()

        def norm_feat(val, min_val, max_val):
            return (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5

        df['temp_score'] = norm_feat(temperature, temp_min, temp_max)
        df['precip_score'] = 1 - (precipitation / precip_max)
        df['pressure_score'] = norm_feat(pressure, pressure_min, pressure_max)

        # Weather composite score
        df['weather_score'] = (df['temp_score'] + df['precip_score'] + df['pressure_score']) / 3

        # Final composite score
        df['pfz_composite_score'] = (
            0.4 * df['ocean_score'] +
            0.3 * df['front_score'] * df['persistence_score'] +
            0.3 * df['weather_score']
        )

        # Now select features for prediction
        try:
            X = df[feature_names]
        except KeyError as e:
            missing_cols = list(set(feature_names) - set(df.columns))
            raise RuntimeError(f"Missing features for prediction: {missing_cols}")

        # Scale features
        X_scaled = scaler.transform(X)

        # Predict class probabilities
        class_probs = model.predict_proba(X_scaled)
        class_preds = model.predict(X_scaled)
        class_map = {
            0: "Low potential zone",
            1: "Medium potential zone",
            2: "High potential zone"
        }
        print(f"Predicted probabilities: {class_probs}")
        print(f"Predicted class: {class_preds}")

        confidence = np.max(class_probs[0]) * 100
        predicted_class = int(class_preds[0])

        explanation = generate_explanation(predicted_class, confidence)
        potential_str = class_map.get(predicted_class, "Unknown")
        
        return {
            "predicted_class": potential_str,
            "confidence_score": confidence,
            "explanation": explanation
        }

def generate_explanation(predicted_class, confidence):
    print(f"Generating explanation for class {predicted_class} with confidence {confidence}")
    if predicted_class == 2:
        return f"High potential fishing zone. Favorable conditions including strong oceanographic signals."
    elif predicted_class == 1:
        return f"Medium potential zone. Moderate environmental signals."
    else:
        return f"Low potential zone. Less favorable conditions."

# def generate_explanation(predicted_class, confidence):
#     print(f"Generating explanation for class {predicted_class} with confidence {confidence}")
#     if predicted_class == 2:
#         return f"High potential fishing zone (confidence: {confidence:.1f}%). Favorable conditions including strong oceanographic signals."
#     elif predicted_class == 1:
#         return f"Medium potential zone (confidence: {confidence:.1f}%). Moderate environmental signals."
#     else:
#         return f"Low potential zone (confidence: {confidence:.1f}%). Less favorable conditions."