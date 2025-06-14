from django.contrib import admin
from django.urls import path,include
from users.views import CreateUserView
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('users/user/register/',CreateUserView.as_view(),name="register"),
    path('users/token/',TokenObtainPairView.as_view(),name="get_token"),
    path('users/token/refresh/',TokenRefreshView.as_view(),name="refresh"),
    path('users-auth/',include("rest_framework.urls")),
    path('quality-assurance/',include("quality_assurance.urls")),
    path('inventory/',include("inventory_management.urls")),
    path('price-predictions/',include("predictions.urls")),
    path('fishing-insights/',include("fishing_insights.urls")),
    path('forecasting/',include("demand_forecasting.urls"))
]
