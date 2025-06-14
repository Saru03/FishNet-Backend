from django.urls import path
from .views import FishPriceForecastView

urlpatterns = [
    path('forecast/',FishPriceForecastView.as_view(), name='forecast'),
]