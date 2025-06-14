from django.urls import path
from .views import PFZView,MapsView,WeatherView

urlpatterns = [
    path('pfz/', PFZView.as_view(), name='pfz-endpoint'),
    path('maps/', MapsView.as_view(), name='maps'),
    path('weather/',WeatherView.as_view(),name='weather')
]