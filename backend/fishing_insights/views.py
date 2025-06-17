from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.parsers import MultiPartParser, FormParser
import datetime
from django.http import JsonResponse
import urllib.parse
from .utils import ProcessedData
from django.http import JsonResponse
import urllib.parse
import requests
import os 

API_KEY = os.environ.get('MAPS_API_KEY')
from .ml_loader import get_ml_components

class PFZView(APIView):
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        # Existing parameter extraction and validation...
        date_str = request.data.get('date')
        latitude_str = request.data.get('latitude')
        longitude_str = request.data.get('longitude')

        if not all([date_str, latitude_str, longitude_str]):
            return Response({"error": "Missing parameters. Please provide date, latitude, longitude."}, status=400)

        try:
            datetime.datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return Response({"error": "Invalid date format. Use YYYY-MM-DD."}, status=400)

        try:
            latitude = float(latitude_str)
            longitude = float(longitude_str)
        except ValueError:
            return Response({"error": "Invalid latitude or longitude. Must be numeric."}, status=400)

        # Lazy load ML components
        model, scaler, rf_imputer = get_ml_components()

        # Pass these models or your data to your fetch_data or prediction logic
        result = ProcessedData.fetch_data(
            date_str, latitude, longitude, 
            model=model, scaler=scaler, rf_imputer=rf_imputer
        )

        if "error" in result:
            return Response(result, status=400)
        else:
            return Response(result, status=200)
        
class MapsView(APIView):
    def get(self, request):
        location = request.GET.get('location', '')
        if not location:
            return JsonResponse({'error': 'No location provided'}, status=400)
        
        encoded_location = urllib.parse.quote_plus(location)
        links = {
            'boat_rentals': f"https://www.google.com/maps/search/boat+rentals+near+{encoded_location}",
            'fish_markets': f"https://www.google.com/maps/search/fish+markets+near+{encoded_location}",
            'cold_storage': f"https://www.google.com/maps/search/cold+storage+facilities+near+{encoded_location}",
            'fishing_gear': f"https://www.google.com/maps/search/fishing+gear+suppliers+near+{encoded_location}",
            'boat_repair': f"https://www.google.com/maps/search/boat+repair+suppliers+near+{encoded_location}",
            'ice_supply': f"https://www.google.com/maps/search/ice+suppliers+near+{encoded_location}",
            'boat_fuel': f"https://www.google.com/maps/search/boat+fuel+suppliers+near+{encoded_location}",
            'fish_packaging': f"https://www.google.com/maps/search/fish+packaging+suppliers+near+{encoded_location}",
        }
        return JsonResponse(links)
    
class WeatherView(APIView):
    def get(self, request):
        location = request.GET.get('location', '')
        if not location:
            return JsonResponse({'error': 'No location provided'}, status=400)

        encoded_location = urllib.parse.quote_plus(location)
        url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={encoded_location}"

        try:
            response = requests.get(url)
            data = response.json()

            # Extract all needed info
            current = data.get('current', {})
            forecast_day = data.get('forecast', {}).get('forecastday', [{}])[0]

            weather_info = {
                'location': location,
                'condition': current.get('condition', {}).get('text', 'N/A'),
                'temp_c': current.get('temp_c', 'N/A'),
                'cloud': current.get('cloud', 'N/A'),
                'humidity': current.get('humidity', 'N/A'),
                'wind_kph': current.get('wind_kph', 'N/A'),
                'gust_kph': current.get('gust_kph', 'N/A'),
                'precip_mm': current.get('precip_mm', 'N/A'),
                'pressure_mb': current.get('pressure_mb', 'N/A'),
                'vis_km': current.get('vis_km', 'N/A'),
                'sunrise': forecast_day.get('astro', {}).get('sunrise', 'N/A'),
                'sunset': forecast_day.get('astro', {}).get('sunset', 'N/A'),
                'moon_phase': forecast_day.get('astro', {}).get('moon_phase', 'N/A'),
                'moon_illumination': forecast_day.get('astro', {}).get('moon_illumination', 'N/A'),
            }
            return JsonResponse(weather_info)

        except Exception as e:
            return JsonResponse({'error': 'Failed to fetch weather data'}, status=500)