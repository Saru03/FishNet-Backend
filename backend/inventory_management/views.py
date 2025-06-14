from rest_framework import generics, status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.db.models import Sum, F, Count, Avg
from django.utils.dateparse import parse_date
from datetime import datetime, date, timedelta
from django.db import transaction
from rest_framework import permissions
from .models import FishStock, Order, OrderItem, Sales, SaleItem
from .serializers import (
    FishStockSerializer,
    OrderSerializer,
    OrderItemSerializer,
    SalesSerializer,
    SaleItemSerializer
)
from rest_framework.parsers import JSONParser, FormParser, MultiPartParser


class FishStockListView(generics.ListCreateAPIView):
    serializer_class = FishStockSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        qs = FishStock.objects.filter(user=self.request.user)
        print("Allowed fish IDs for user:", list(qs.values_list('id', flat=True)))
        return qs

    def get_serializer_context(self):
        return {'request': self.request}

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class FishStockDetailView(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = FishStockSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return FishStock.objects.filter(user=self.request.user)

    def get_serializer_context(self):
        return {'request': self.request}


class OrderListView(generics.ListCreateAPIView):
    serializer_class = OrderSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        converted_order_ids = Sales.objects.filter(
            source='order', order__isnull=False
        ).values_list('order_id', flat=True)

        return Order.objects.filter(user=self.request.user).exclude(id__in=converted_order_ids).prefetch_related('items')

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['request'] = self.request
        return context

    def patch(self, request, *args, **kwargs):
        # Handle PATCH requests for updating order status
        order_id = request.data.get('order_id')
        if not order_id:
            return Response({'error': 'order_id is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            instance = Order.objects.get(id=order_id, user=request.user)
            old_status = instance.status
            
            print(f"Updating order {order_id} from {old_status} to {request.data.get('status')}")
            
            # Update the order status first
            new_status = request.data.get('status')
            instance.status = new_status
            instance.save()
            
            print(f"Order {order_id} status updated successfully to {new_status}")
            
            # Check if status changed to completed
            if old_status != 'completed' and new_status == 'completed':
                print(f"Converting order {order_id} to sale...")
                
                # Auto-move to sales
                try:
                    with transaction.atomic():
                        # Get order items BEFORE any deletion
                        order_items = list(instance.items.select_related('fish').all())
                        print(f"Found {len(order_items)} items in order {order_id}")
                        
                        if not order_items:
                            return Response({
                                'error': 'No items found in order'
                            }, status=status.HTTP_400_BAD_REQUEST)
                        
                        # Prepare item_details for sale
                        item_details = []
                        for item in order_items:
                            if not item.fish:
                                print(f"Warning: Order item {item.id} has no fish reference")
                                continue
                                
                            item_detail = {
                                'fish_name': item.fish.name,
                                'quantity': item.quantity,
                                'price_per_unit': float(item.price_per_unit)  # Ensure it's a float
                            }
                            item_details.append(item_detail)
                            print(f"Added item: {item_detail}")
                        
                        if not item_details:
                            return Response({
                                'error': 'No valid items found in order'
                            }, status=status.HTTP_400_BAD_REQUEST)
                        
                        # Create sale data with proper formatting
                        sale_data = {
                            'sale_date': datetime.now().isoformat(),
                            'source': 'order',
                            'order': instance.id,
                            'customer_name': instance.customer_name or '',
                            'item_details': item_details,
                        }
                        
                        print(f"Sale data prepared: {sale_data}")
                        
                        # Create sale using serializer with proper context
                        context = {'request': request, 'source': 'order'}
                        sales_serializer = SalesSerializer(data=sale_data, context=context)
                        
                        if sales_serializer.is_valid():
                            sale = sales_serializer.save()
                            print(f"✓ Order {instance.id} successfully converted to sale {sale.id}")
                            
                            # Verify the sale was created properly
                            created_sale = Sales.objects.get(id=sale.id)
                            print(f"✓ Sale verification - ID: {created_sale.id}, Order: {created_sale.order_id}, Items: {created_sale.items.count()}")
                            
                            # Return success response with updated order
                            serializer = self.get_serializer(instance)
                            return Response({
                                'message': 'Order completed and converted to sale successfully',
                                'order': serializer.data,
                                'sale_id': sale.id
                            })
                            
                        else:
                            print(f"✗ Failed to create sale: {sales_serializer.errors}")
                            # Rollback order status change
                            instance.status = old_status
                            instance.save()
                            return Response({
                                'error': 'Failed to convert order to sale',
                                'details': sales_serializer.errors
                            }, status=status.HTTP_400_BAD_REQUEST)
                            
                except Exception as e:
                    print(f"✗ Error auto-moving order to sales: {e}")
                    import traceback
                    traceback.print_exc()
                    # Rollback order status change
                    instance.status = old_status
                    instance.save()
                    return Response({
                        'error': 'Failed to convert order to sale',
                        'details': str(e)
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # For non-completed status changes, just return the updated order
            serializer = self.get_serializer(instance)
            return Response(serializer.data)
                
        except Order.DoesNotExist:
            print(f"✗ Order {order_id} not found")
            return Response({'error': 'Order not found'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class OrderDetailView(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = OrderSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Order.objects.filter(user=self.request.user).prefetch_related('items')

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['request'] = self.request
        return context
    

class SalesListView(generics.ListCreateAPIView):
    serializer_class = SalesSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        queryset = Sales.objects.filter(user=self.request.user).prefetch_related('items__fish').order_by('-sale_date')
        print(f"Sales queryset count: {queryset.count()}")
        for sale in queryset:
            print(f"Sale ID: {sale.id}, Source: {sale.source}, Order: {sale.order_id}, Items: {sale.items.count()}")
        return queryset

    def get_serializer_context(self):
        return {'request': self.request}


class SalesDetailView(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = SalesSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Sales.objects.filter(user=self.request.user).prefetch_related('items__fish')

    def get_serializer_context(self):
        return {'request': self.request}
      
# Stock report view
class StockReportView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        stocks = FishStock.objects.filter(user=request.user)

        stock_data = stocks.values('category').annotate(
            total_quantity=Sum('quantity'),
            total_items=Count('id'),
            avg_price=Avg('price')
        )

        low_stock_items = stocks.filter(quantity__lt=F('low_stock_threshold'))

        stock_value = sum(item.quantity * item.price for item in stocks)

        total_fish_count = stocks.aggregate(total=Sum('quantity'))['total'] or 0
        unique_species = stocks.count()

        top_valuable_items = stocks.annotate(
            total_value=F('quantity') * F('price')
        ).order_by('-total_value')[:5]

        return Response({
            'stock_data': list(stock_data),
            'low_stock_items': FishStockSerializer(low_stock_items, many=True, context={'request': request}).data,
            'stock_value': round(stock_value, 2),
            'total_fish_count': total_fish_count,
            'unique_species': unique_species,
            'top_valuable_items': FishStockSerializer(top_valuable_items, many=True, context={'request': request}).data,
            'low_stock_count': low_stock_items.count()
        })

# Sales report
class SalesReportView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    def get(self, request):
        try:
            start_date_str = request.GET.get('start_date')
            end_date_str = request.GET.get('end_date')

            start_date = parse_date(start_date_str) if start_date_str else None
            end_date = parse_date(end_date_str) if end_date_str else None

            if start_date_str and not start_date:
                return Response({'error': 'Invalid start_date format. Use YYYY-MM-DD'}, status=status.HTTP_400_BAD_REQUEST)
            if end_date_str and not end_date:
                return Response({'error': 'Invalid end_date format. Use YYYY-MM-DD'}, status=status.HTTP_400_BAD_REQUEST)

            sales_qs = Sales.objects.filter(user=request.user)
            sale_items_qs = SaleItem.objects.filter(sale__user=request.user)

            if start_date and end_date:
                sales_qs = sales_qs.filter(sale_date__date__range=[start_date, end_date])
                sale_items_qs = sale_items_qs.filter(sale__sale_date__date__range=[start_date, end_date])
            elif start_date:
                sales_qs = sales_qs.filter(sale_date__date__gte=start_date)
                sale_items_qs = sale_items_qs.filter(sale__sale_date__date__gte=start_date)
            elif end_date:
                sales_qs = sales_qs.filter(sale_date__date__lte=end_date)
                sale_items_qs = sale_items_qs.filter(sale__sale_date__date__lte=end_date)

            total_revenue = sale_items_qs.aggregate(
                total=Sum(F('quantity') * F('price_per_unit'))
            )['total'] or 0

            total_sales = sales_qs.count()
            total_items_sold = sale_items_qs.aggregate(
                total=Sum('quantity')
            )['total'] or 0

            average_sale_value = sale_items_qs.aggregate(
                avg=Avg(F('quantity') * F('price_per_unit'))
            )['avg'] or 0

            best_selling_items = sale_items_qs.values(
                'fish__name', 'fish__id'
            ).annotate(
                total_quantity=Sum('quantity'),
                total_revenue=Sum(F('quantity') * F('price_per_unit')),
                sales_count=Count('sale', distinct=True)
            ).order_by('-total_quantity')[:10]

            # For daily trend
            if not start_date:
                start_date = datetime.now().date() - timedelta(days=7)
            if not end_date:
                end_date = datetime.now().date()

            daily_sales = Sales.objects.filter(sale_date__date__range=[start_date, end_date]).extra(
                select={'day': 'DATE(sale_date)'}
            ).values('day').annotate(
                daily_revenue=Sum(F('items__quantity') * F('items__price_per_unit')),
                daily_sales_count=Count('id'),
                daily_items_sold=Sum('items__quantity')
            ).order_by('day')

            recent_sales = sales_qs.order_by('-sale_date')[:10]
            recent_sales_data = SalesSerializer(recent_sales, many=True, context={'request': request}).data

            return Response({
                'summary': {
                    'total_revenue': float(total_revenue),
                    'total_sales': total_sales,
                    'total_items_sold': total_items_sold,
                    'average_sale_value': float(average_sale_value),
                },
                'best_selling_items': list(best_selling_items),
                'daily_sales_trend': list(daily_sales),
                'recent_sales': recent_sales_data,
                'date_range': {
                    'start_date': start_date.isoformat() if start_date else None,
                    'end_date': end_date.isoformat() if end_date else None
                }
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Order report with status count
class OrderReportView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    def get(self, request):
        try:
            total_orders = Order.objects.filter(user=request.user).count()
            completed_orders = Order.objects.filter(user=request.user, status='completed').count()
            pending_orders = Order.objects.filter(user=request.user, status='pending').count()
            canceled_orders = Order.objects.filter(user=request.user, status='cancelled').count()

            orders_by_status = Order.objects.filter(user=request.user).values('status').annotate(total=Count('id'))

            return Response({
                'orders_by_status': list(orders_by_status),
                'summary': {
                    'total_orders': total_orders,
                    'completed_orders': completed_orders,
                    'pending_orders': pending_orders,
                    'cancelled_orders': canceled_orders,
                    'completion_rate': round((completed_orders / total_orders * 100), 2) if total_orders > 0 else 0
                }
            })
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Orders by date with detailed analytics
class OrdersByDateReportView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    def get(self, request):
        try:
            start_date_str = request.GET.get('start_date')
            end_date_str = request.GET.get('end_date')

            start_date = parse_date(start_date_str) if start_date_str else None
            end_date = parse_date(end_date_str) if end_date_str else None

            # Validate
            if start_date_str and not start_date:
                return Response({'error': 'Invalid start_date format. Use YYYY-MM-DD'}, status=status.HTTP_400_BAD_REQUEST)
            if end_date_str and not end_date:
                return Response({'error': 'Invalid end_date format. Use YYYY-MM-DD'}, status=status.HTTP_400_BAD_REQUEST)

            # Validate date range
            if start_date and end_date and start_date > end_date:
                return Response({'error': 'start_date cannot be later than end_date'}, status=status.HTTP_400_BAD_REQUEST)

            # Filter orders
            qs = Order.objects.filter(user=request.user)
            if start_date and end_date:
                qs = qs.filter(order_date__range=[start_date, end_date])
                date_range_str = f"{start_date} to {end_date}"
            elif start_date:
                qs = qs.filter(order_date__gte=start_date)
                date_range_str = f"From {start_date}"
            elif end_date:
                qs = qs.filter(order_date__lte=end_date)
                date_range_str = f"Until {end_date}"
            else:
                date_range_str = "All time"

            orders_serialized = OrderSerializer(qs, many=True, context={'request': request}).data
            total_orders = qs.count()

            orders_by_status = qs.values('status').annotate(total=Count('id'))

            daily_breakdown = qs.extra(
                select={'day': 'date(order_date)'}
            ).values('day').annotate(
                count=Count('id'),
                total_value=Sum(F('items__quantity') * F('items__price_per_unit'))
            ).order_by('day')

            return Response({
                'date_range': date_range_str,
                'total_orders': total_orders,
                'orders': orders_serialized,
                'orders_by_status': list(orders_by_status),
                'daily_breakdown': list(daily_breakdown),
            })

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Order analytics
class OrderAnalyticsView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    def get(self, request):
        try:
            today = date.today()
            current_month_start = today.replace(day=1)
            if current_month_start.month == 1:
                last_month_start = current_month_start.replace(year=current_month_start.year - 1, month=12)
            else:
                last_month_start = current_month_start.replace(month=current_month_start.month - 1)
            current_month_end = (current_month_start + timedelta(days=31)).replace(day=1) - timedelta(days=1)
            last_month_end = current_month_start - timedelta(days=1)

            # Orders in current month
            current_month_orders = Order.objects.filter(
                user=request.user,
                order_date__gte=current_month_start,
                order_date__lte=current_month_end
            ).count()

            # Last month orders
            last_month_orders = Order.objects.filter(
                user=request.user,
                order_date__gte=last_month_start,
                order_date__lte=last_month_end
            ).count()

            # Growth
            growth_pct = 0
            if last_month_orders > 0:
                growth_pct = round(((current_month_orders - last_month_orders) / last_month_orders) * 100, 2)

            return Response({
                'current_month_orders': current_month_orders,
                'last_month_orders': last_month_orders,
                'growth_percentage': growth_pct,
                'month_name': current_month_start.strftime('%B %Y')
            })

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)