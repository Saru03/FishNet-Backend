from django.urls import path
from .views import (
    FishStockListView,
    FishStockDetailView,
    OrderListView,
    OrderDetailView,
    SalesListView,
    SalesDetailView,
    StockReportView,
    SalesReportView,
    OrderReportView,
    OrdersByDateReportView,
    OrderAnalyticsView
)

urlpatterns = [
    path('fishstock/', FishStockListView.as_view(), name='fishstock-list'),
    path('fishstock/<int:pk>/', FishStockDetailView.as_view(), name='fishstock-detail'),
    path('orders/', OrderListView.as_view(), name='order-list'),
    path('orders/<int:pk>/', OrderDetailView.as_view(), name='order-detail'),
    path('sales/', SalesListView.as_view(), name='sales-list'),
    path('sales/<int:pk>/', SalesDetailView.as_view(), name='sales-detail'),
    # Reports
    path('stock-report/', StockReportView.as_view(), name='stock-report'),
    path('sales-report/', SalesReportView.as_view(), name='sales-report'),
    path('orders/report/', OrderReportView.as_view(), name='order-report'),
    path('orders/report/by-date/', OrdersByDateReportView.as_view(), name='orders-by-date'),
    path('orders/analytics/', OrderAnalyticsView.as_view(), name='order-analytics'),
]