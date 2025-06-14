from django.urls import path
from .views import FreshnessCheckView, DiseaseDetectionView

urlpatterns = [
    path('quality-check/freshness/', FreshnessCheckView.as_view(), name='freshness-check'),
    path('quality-check/disease/', DiseaseDetectionView.as_view(), name='disease-check'),
]
