from django.http import JsonResponse
from django.views.decorators.http import require_GET
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from .utils import ForecastModelLoader # Only import the ForecastModelLoader

class FishPriceForecastView(APIView):
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        fore_date_str = request.GET.get('date')
        market = request.GET.get('market')
        fish = request.GET.get('fish')
        size = request.GET.get('size')

        if not all([fore_date_str, market, fish, size]):
            return Response({"error": "Missing parameters. Please provide date, market, fish, and size."}, status=400)

        result = ForecastModelLoader.predict_fish_price(fore_date_str, market, fish, size)

        if "error" in result:
            return Response(result, status=400)
        else:
            return Response(result, status=200)