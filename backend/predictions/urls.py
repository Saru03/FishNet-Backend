from django.urls import path
from .views import  MarketComparisonView

urlpatterns = [
    path('compare-markets/',MarketComparisonView.as_view(), name="compare-markets")
]