import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from datetime import datetime, timedelta
from geopy.distance import geodesic
import geopy.geocoders
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import RobustScaler
from django.conf import settings
from .mappings import fish_mapping, region, market_mapping, market_locations

class ModelLoader:
    model_ann = None

    @classmethod
    def load_model(cls):
        if cls.model_ann is None:
            model_path = os.path.join(settings.BASE_DIR, 'ml-models', 'model_ann.pkl')
            with open(model_path, 'rb') as f:
                cls.model_ann = pickle.load(f)

def get_distance(user_location, market_location):
    return geodesic(user_location, market_location).km

def get_coordinates(location_name):
    gl = geopy.geocoders.Nominatim(user_agent="geoapi")
    try:
        location = gl.geocode(location_name)
        return (location.latitude, location.longitude) if location else None
    except:
        return None

class MarketComparisonView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            fish = request.data.get("fish")
            size = request.data.get("size")
            user_location = request.data.get("location")
            date_str = request.data.get("date")

            if not all([fish, size, user_location, date_str]):
                return Response({"error": "Missing input fields"}, status=400)

            fish = fish_mapping.get(fish, fish)
            fore_date = datetime.strptime(date_str, "%Y-%m-%d")
            user_coords = get_coordinates(user_location)
            if not user_coords:
                return Response({"error": "Invalid user location"}, status=400)

            train_data_path = os.path.join(settings.BASE_DIR, 'predictions', 'all_states_aggregate.xlsx')
            train_data = pd.read_excel(train_data_path)
            train_data.dropna(subset=['Date'], inplace=True)
            train_data['Date'] = pd.to_datetime(train_data['Date'], format='%d-%m-%Y', errors='coerce')


            full_dates = pd.DataFrame({'Date': pd.date_range(start=train_data['Date'].min(), end=train_data['Date'].max(), freq='D')})
            train_data = full_dates.merge(train_data, on='Date', how='left')
            train_data.set_index('Date', inplace=True)
            train_data = train_data.interpolate(method="linear", limit_direction="both")
            train_data.replace(r'^\s*$', np.nan, regex=True, inplace=True)
            train_data = train_data.apply(pd.to_numeric, errors='coerce')
            for col in train_data.columns:
                train_data[col] = train_data[col].fillna(train_data[col].median())
                if (train_data[col] < 0).any():
                    mean_val = train_data.loc[train_data[col] > 0, col].mean()
                    train_data[col] = train_data[col].apply(lambda x: x if x > 0 else mean_val)

            scaler = RobustScaler()
            train_scaled = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns, index=train_data.index)

            baseline_date = datetime.strptime("07-03-2025", "%d-%m-%Y")
            if fore_date < baseline_date:
                return Response({"error": "Date must be after 7th March 2025"}, status=400)

            ModelLoader.load_model()
            model = ModelLoader.model_ann

            steps = (fore_date - baseline_date).days
            input_data = train_scaled.values[-90:].reshape(1, 90, -1)
            forecast_scaled = []

            for _ in range(steps):
                pred = model.predict(input_data)
                forecast_scaled.append(pred[0])
                new_input = np.concatenate([input_data[0][1:], pred], axis=0)
                input_data = new_input.reshape(1, 90, -1)

            forecast = scaler.inverse_transform(forecast_scaled)
            forecast_dates = [baseline_date + timedelta(days=i + 1) for i in range(steps)]
            forecast_data = pd.DataFrame(forecast, columns=train_data.columns, index=forecast_dates)

            market_predictions = []
            for market, coords in market_locations.items():
                fish_col = f"{fish}_{size}_Market{market_mapping[market]}"
                if fish_col in forecast_data.columns:
                    price = np.round(forecast_data.loc[fore_date, fish_col])
                    distance = get_distance(user_coords, coords)
                    maps_url = f"https://www.google.com/maps/dir/{user_coords[0]},{user_coords[1]}/{coords[0]},{coords[1]}"
                    market_predictions.append({
                        "region": region[market],
                        "market": market,
                        "predicted_price": price,
                        "distance_km": round(distance, 2),
                        "route_url": maps_url
                    })
                else:
                    distance = get_distance(user_coords, coords)
                    market_predictions.append({
                        "region": region[market],
                        "market": market,
                        "predicted_price": "Not Sold Here",
                        "distance_km": round(distance, 2),
                        "route_url": None
                    })

            market_predictions.sort(key=lambda x: x["distance_km"])
            return Response({"results": market_predictions})

        except Exception as e:
            return Response({"error": str(e)}, status=500)
