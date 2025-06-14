# myapp/utils.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
import pickle
import traceback
from django.conf import settings # Import settings
import matplotlib.pyplot as plt

EXCEL_FILE_PATH = os.path.join(settings.BASE_DIR, 'demand_forecasting', 'all_states_aggregate.xlsx') # Use settings.BASE_DIR
MODEL_FILE_PATH = os.path.join(settings.BASE_DIR, 'ml-models', 'forecast.pkl') 

def get_fish_column(fish, size, market):
    fish_mapping = {
       'Indian Mackerel': 'Indian_Mackerel',
       'Rock Lobster': 'Rock_Lobster',
       'Silver Pomfret': 'Silver_Pomfret',
       'Indian White Prawn':'Indian_White_Prawn',
       'Spotted Crab':'Spotted_Crab',
       'Indo Pacific Seer Fish':'Indo_Pacific_Seer_Fish',
       'Johns Snapper':'Johns_Snapper'
    }
    size_mapping = {
        "Small": "Small",
        "Medium": "Medium",
        "Large": "Large"
    }
    market_mapping = {
        "Mohanpura Modern Fish Market": "MarketA",
        "Venlok Fish Market": "MarketB",
        "Visakhapatnam Municipal Wholesale Fish Market": "MarketC",
        "Ghoghla Retail Fish Market": "MarketD",
        "Vanakbara Retail Fish Market": "MarketE",
        "Mapusa Fish Market": "MarketF",
        "SGPDA Wholesale Fish Market": "MarketG",
        "Madikeri Retail Fish Market": "MarketH",
        "Padubidri Retail Fish Market":"MarketI",
        "Kavanad Retail Fish Market":"MarketJ",
        "Ponnani Harbour Azheekkal":"MarketK",
        "Panvel Fish Market":"MarketL",
        "Sasoon Dock Retail Fish Market":"MarketM",
        "Naya Bazar Retail Fish Market":"MarketN",
        "Unit-4 Wholesale Fish Market":"MarketO",
        "Badagada Retail Fish Market":"MarketP",
        "Chintadripet Fish Market":"MarketQ",
        'Kasivillangi Fish Market': 'MarketR',
        'Nelpattai Fish Market': 'MarketS',
    }

    if fish in fish_mapping and size in size_mapping and market in market_mapping:
        return f"{fish_mapping[fish]}_{size_mapping[size]}_{market_mapping[market]}"
    else:
        return "Incorrect Input"

def load_and_prepare_data():
    try:
        train_data = pd.read_excel(EXCEL_FILE_PATH)
        train_data.dropna(subset=['Date'], inplace=True)
        train_data['Date'] = pd.to_datetime(train_data['Date'], format='%d-%m-%Y', errors='coerce')
        full_dates = pd.DataFrame({'Date': pd.date_range(start=train_data['Date'].min(), end=train_data['Date'].max(), freq='D')})
        train_data = full_dates.merge(train_data, on='Date', how='left')
        train_data.set_index('Date', inplace=True)
        for col in train_data.columns:
            train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
        train_data.replace(0, np.nan, inplace=True)
        train_data = train_data.interpolate(method="linear", limit_direction="both")
        for col in train_data.columns:
            train_data[col] = train_data[col].fillna(train_data[col].median())
        for col in train_data.columns:
            if (train_data[col] < 0).any():
                smooth_mean = train_data.loc[train_data[col] > 0, col].mean()
                train_data[col] = train_data[col].apply(lambda x: x if x > 0 else smooth_mean)
        return train_data
    except FileNotFoundError:
        print(f"Error: Excel file not found at {EXCEL_FILE_PATH}")
        return None
    except Exception as e:
        print(f"Error loading and preparing data: {e}")
        traceback.print_exc()
        return None

class ForecastModelLoader:
    model = None
    scaler = None
    train_data = None

    @classmethod
    def load_model_and_data(cls):
        if cls.model is None or cls.scaler is None or cls.train_data is None:
            print("Loading forecasting model, scaler, and data...") # Add a print statement to see when loading happens
            # Load data first
            cls.train_data = load_and_prepare_data()
            if cls.train_data is None:
                print("Failed to load training data.")
                return False # Indicate failure to load data

            # Initialize and fit scaler
            try:
                cls.scaler = RobustScaler()
                cls.scaler.fit(cls.train_data)  # Fit the scaler on the entire training data
                print("Scaler fitted successfully.")
            except Exception as e:
                print(f"Error fitting scaler: {e}")
                traceback.print_exc()
                cls.scaler = None
                cls.train_data = None # Reset data if scaler fails
                return False

            # Load the model using pickle
            try:
                with open(MODEL_FILE_PATH, 'rb') as file:
                    cls.model = pickle.load(file)
                print("Forecasting model loaded successfully.")
                return True # Indicate success
            except FileNotFoundError:
                print(f"Error: Model file not found at {MODEL_FILE_PATH}")
                cls.model = None
                cls.scaler = None # Reset scaler if model fails
                cls.train_data = None # Reset data if model fails
                return False # Indicate failure
            except Exception as e:
                print(f"Error loading model: {e}")
                traceback.print_exc()
                cls.model = None
                cls.scaler = None # Reset scaler if model fails
                cls.train_data = None # Reset data if model fails
                return False
        else:
            print("Forecasting model, scaler, and data already loaded (cached).")
            return True # Indicate that they were already loaded
            
    @staticmethod
    def plot_fish_price_forecast(train_data, forecast_dates, fish_and_size, fish_data):
        """
        Plots historical and forecasted fish prices.

        Parameters:
        - train_data: DataFrame containing historical data with Date index
        - forecast_dates: list of datetime objects for forecasted dates
        - fish_and_size: string, column name for the specific fish and size
        - fish_data: Series or array of forecasted values
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_data.index, train_data[fish_and_size], label="Historical Data", color="blue")
        plt.plot(forecast_dates, fish_data, label="Forecast Data", linestyle="--", color="red")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title(f"Fish Price Forecast for {fish_and_size}")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    @classmethod
    def predict_fish_price(cls, fore_date_str, market, fish, size):
        if not cls.load_model_and_data():
            return {"error": "Model, scaler, or data failed to load. Please check your setup and server logs."}

        try:
            fore_date = datetime.strptime(fore_date_str, "%Y-%m-%d")  # Assuming input format is YYYY-MM-DD
        except ValueError:
            return {"error": "Invalid date format. Please use YYYY-MM-DD."}

        baseline_date = datetime.strptime("07-03-2025", "%d-%m-%Y")

        if fore_date < baseline_date:
            return {"error": "Enter dates after 7th March, 2025 only."}

        steps = (fore_date - baseline_date).days

        # Scale the last 90 days of the training data
        train_scaled = pd.DataFrame(cls.scaler.transform(cls.train_data),
                                    columns=cls.train_data.columns,
                                    index=cls.train_data.index)

        input_data = train_scaled.values[-90:].reshape(1, 90, -1)
        forecast_scaled = []

        for _ in range(steps):
            pred = cls.model.predict(input_data)
            forecast_scaled.append(pred[0])
            new_input = np.concatenate([input_data[0][1:], pred], axis=0)
            input_data = new_input.reshape(1, 90, -1)

        forecast = cls.scaler.inverse_transform(forecast_scaled)
        baseline_date_forecast = baseline_date + timedelta(days=1)  # Start forecast from the day after baseline
        forecast_dates = [baseline_date_forecast + timedelta(days=i) for i in range(steps)]

        # Get the column name for the specific fish, size, and market
        fish_and_size = get_fish_column(fish, size, market)
        if fish_and_size == "Incorrect Input":
            return {"error": "Incorrect fish, size, or market."}

        # Plot forecast
        try:
            cls.plot_fish_price_forecast(cls.train_data, forecast_dates, fish_and_size, forecast)
        except Exception as e:
            print(f"Error during plotting: {e}")

        # Prepare forecast data for JSON response
        forecast_data = pd.DataFrame(forecast, columns=cls.train_data.columns, index=forecast_dates)

        if fish_and_size not in forecast_data.columns:
            return {"error": f"Data for {fish} ({size}, {market}) not found in forecast."}

        fish_data = forecast_data[fish_and_size]

        forecast_results = []
        for date, prediction in fish_data.items():
            forecast_results.append({
                'date': date.strftime("%Y-%m-%d"),
                'prediction': np.round(prediction)
            })

        return {"forecast": forecast_results}
        
         
   