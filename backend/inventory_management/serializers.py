from rest_framework import serializers
from .models import FishStock, Order, OrderItem, Sales, SaleItem

class FishStockSerializer(serializers.ModelSerializer):
    class Meta:
        model = FishStock
        fields = '__all__'
        read_only_fields = ['user']  # user set in view or validated


class OrderItemSerializer(serializers.ModelSerializer):
    fish = FishStockSerializer(read_only=True)
    fish_name = serializers.CharField(write_only=True)
    price_per_unit = serializers.DecimalField(max_digits=10, decimal_places=2)

    class Meta:
        model = OrderItem
        fields = ['id', 'fish', 'fish_name', 'quantity', 'price_per_unit']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        request = self.context.get('request', None)
        if request and getattr(request, 'user', None):
            pass

    def validate(self, data):
        fish_name = data.get('fish_name')
        print(f"Looking up fish with name: '{fish_name}'")
        quantity = data.get('quantity')
        request = self.context.get('request')
        user = request.user if request else None

        if not fish_name:
            raise serializers.ValidationError("Fish name is required.")

        try:
            fish = FishStock.objects.get(name__iexact=fish_name.strip(),user=request.user)
            print(f"Found fish: {fish.name}") 
        except FishStock.DoesNotExist:
            print("Fish not found") 
            raise serializers.ValidationError(f"Fish '{fish_name}' not found.")

        if request:
            if quantity > fish.quantity:
                raise serializers.ValidationError(f"Not enough stock for {fish.name}. Available: {fish.quantity}")

        data['fish'] = fish
        return data

class OrderSerializer(serializers.ModelSerializer):
    items = OrderItemSerializer(many=True, read_only=True)
    item_details = OrderItemSerializer(required=True, many=True, write_only=True)
    order_date = serializers.DateField(read_only=True)

    class Meta:
        model = Order
        fields = [
            'id', 'order_date', 'delivery_date', 'status',
            'customer_name', 'items', 'item_details'
        ]

    def create(self, validated_data):
        item_details = validated_data.pop('item_details') 
        request = self.context['request']
        user = request.user

        order = Order.objects.create(user=user, **validated_data)

        for item in item_details:
            fish = item.get('fish')
            quantity = item.get('quantity')
            price_per_unit = item.get('price_per_unit')

            if not fish or not quantity or not price_per_unit:
                raise serializers.ValidationError("Missing required fields for orderiems")

            if fish.quantity < quantity:
                raise serializers.ValidationError(f"Not enough stock for {fish.name}.")

            # Reduce stock
            fish.reduce_stock(quantity)

            # Create order item
            order_item = OrderItem.objects.create(
                order=order,
                fish=fish,
                quantity=quantity,
                price_per_unit=price_per_unit
            )
            print(f"OrderItem created:{order_item}")

        return order

    def update(self, instance, validated_data):
        item_details = validated_data.pop('item_details', None)
        request = self.context['request']
        user = request.user

        # Restore stock from existing items
        for item in instance.items.all():
            item.fish.quantity += item.quantity
            item.fish.save()

        instance.items.all().delete()

        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()

        if item_details:
            for item in item_details:
                fish = item.get('fish')
                quantity = item.get('quantity')
                price_per_unit = item.get('price_per_unit')

                if not fish or not quantity or not price_per_unit:
                    raise serializers.ValidationError("Missing required fields for order items.")

                if fish.quantity < quantity:
                    raise serializers.ValidationError(f"Not enough stock for {fish.name}.")

                fish.reduce_stock(quantity)

                OrderItem.objects.create(
                    order=instance,
                    fish=fish,
                    quantity=quantity,
                    price_per_unit=price_per_unit
                )
                print(f"OrderItem created: {fish.name}, Qty: {quantity}")

        return instance  

class SaleItemSerializer(serializers.ModelSerializer):
    fish = FishStockSerializer(read_only=True)
    fish_name = serializers.CharField(write_only=True)
    revenue = serializers.DecimalField(max_digits=10, decimal_places=2, read_only=True)

    class Meta:
        model = SaleItem
        fields = ['id', 'fish', 'fish_name', 'quantity', 'price_per_unit', 'revenue']

    def validate(self, data):
        print(f"SaleItemSerializer.validate called with: {data}")
        
        fish_name = data.get('fish_name')
        quantity = data.get('quantity')
        request = self.context.get('request')
        source = self.context.get('source', 'manual')

        if not fish_name:
            raise serializers.ValidationError("Fish name is required.")

        print(f"Looking for fish: '{fish_name}', source: '{source}'")

        # Find existing fish
        try:
            fish = FishStock.objects.get(name__iexact=fish_name.strip(),user=request.user)
            print(f"Found fish: {fish.name} (ID: {fish.id})")
        except FishStock.DoesNotExist:
            print(f"Fish '{fish_name}' not found")
            
            if source == 'order':
                raise serializers.ValidationError(f"Fish '{fish_name}' not found in stock.")
            
          
            print(f"Creating new fish: {fish_name}")
            fish = FishStock.objects.create(
                name=fish_name.strip(),
                category='Unknown',
                quantity=0,
                price='0.00',
                user=request.user if request else None
            )

        data['fish'] = fish
        print(f"Fish object assigned to data: {fish.name}")

        if request and source == 'manual':
            if quantity > fish.quantity:
                raise serializers.ValidationError(f"Not enough stock for {fish.name}. Available: {fish.quantity}")

        print(f"SaleItemSerializer validation successful: {data.keys()}")
        return data


class SalesSerializer(serializers.ModelSerializer):
    items = SaleItemSerializer(many=True, read_only=True)
    item_details = serializers.ListField(write_only=True)
    sale_date = serializers.DateTimeField()
    source = serializers.CharField()
    customer_name = serializers.CharField(required=False, allow_blank=True)
    order = serializers.PrimaryKeyRelatedField(queryset=Order.objects.all(), required=False, allow_null=True)
    order_id = serializers.PrimaryKeyRelatedField(queryset=Order.objects.all(), source='order', write_only=True, required=False, allow_null=True)

    class Meta:
        model = Sales
        fields = ['id', 'sale_date', 'source', 'order', 'order_id', 'customer_name', 'items', 'item_details']

    def create(self, validated_data):
        print(f"SalesSerializer.create called with data: {validated_data}")
        
        item_details_data = validated_data.pop('item_details', [])
        print(f"Item details extracted: {len(item_details_data)} items")
        print(f"Raw item_details_data: {item_details_data}")
        
        request = self.context['request']
        user = request.user
        source = validated_data.get('source', 'manual')
        
        # Get the order if it exists
        order = validated_data.get('order', None)
        print(f"Order from validated_data: {order}")

        customer_name = validated_data.pop('customer_name', None)
        validated_data.pop('user', None)

        print(f"Creating sale with data: {validated_data}")
        sale = Sales.objects.create(**validated_data, user=user)
        print(f"✓ Sale created with ID: {sale.id}, Order: {sale.order}")

        
        if customer_name:
            try:
                if hasattr(Sales, '_meta'):
                    
                    field_names = [field.name for field in Sales._meta.get_fields()]
                    if 'customer_name' in field_names:
                        sale.customer_name = customer_name
                        sale.save()
                        print(f"✓ Customer name set directly: {sale.customer_name}")
                    else:
                        print(f"⚠ customer_name is not a direct field, might be a property derived from order")
                       
                        if order and hasattr(order, 'customer_name'):
                            print(f"✓ Customer name available from order: {order.customer_name}")
            except Exception as e:
                print(f"⚠ Could not set customer_name directly: {e}")
                

        
        for i, item_data in enumerate(item_details_data):
            print(f"Processing item {i+1}: {item_data}")
            
            
            try:
                fish_name = str(item_data.get('fish_name', '')).strip()
                quantity = int(item_data.get('quantity', 0))
                price_per_unit = float(item_data.get('price_per_unit', 0))
            except (ValueError, TypeError) as e:
                print(f"✗ Data type conversion error for item {i+1}: {e}")
                continue
            
            if not fish_name:
                print(f"✗ No fish_name found in item_data: {item_data}")
                continue
            
            if not quantity or quantity <= 0:
                print(f"✗ Invalid quantity found in item_data: {quantity}")
                continue
                
            if not price_per_unit or price_per_unit <= 0:
                print(f"✗ Invalid price_per_unit found in item_data: {price_per_unit}")
                continue
            
            print(f"✓ Item data validated - Fish: {fish_name}, Quantity: {quantity}, Price: {price_per_unit}")
            
            
            try:
                fish = FishStock.objects.get(name__iexact=fish_name,user=request.user)
                print(f"✓ Found fish: {fish.name} (ID: {fish.id})")
            except FishStock.DoesNotExist:
                print(f"✗ Fish '{fish_name}' not found")
                if source == 'order':
                    raise serializers.ValidationError(f"Fish '{fish_name}' not found in stock.")
                
                
                fish = FishStock.objects.create(
                    name=fish_name,
                    category='Unknown',
                    quantity=0,
                    price='0.00',
                    user=user
                )
                print(f"✓ Created new fish: {fish.name}")
            
            
            if source == 'manual':
                # Only reduce stock for manual sales
                if quantity > fish.quantity:
                    raise serializers.ValidationError(f"Not enough stock for {fish.name}. Available: {fish.quantity}")
                fish.quantity -= quantity
                fish.save()
                print(f"✓ Stock reduced for {fish.name}: -{quantity}")
            else:
                print(f"✓ Skipping stock reduction for order conversion - stock already reduced")

            
            try:
                sale_item = SaleItem.objects.create(
                    sale=sale,
                    fish=fish,
                    quantity=quantity,
                    price_per_unit=price_per_unit
                )
                print(f"✓ SaleItem created successfully with ID: {sale_item.id}")
                
                
                if SaleItem.objects.filter(id=sale_item.id, sale=sale).exists():
                    print(f"✓ SaleItem {sale_item.id} confirmed linked to sale {sale.id}")
                else:
                    print(f"✗ SaleItem {sale_item.id} NOT properly linked to sale {sale.id}!")
                    
            except Exception as e:
                print(f"✗ ERROR creating SaleItem: {e}")
                import traceback
                traceback.print_exc()
                raise

       
        sale.refresh_from_db()
        items_count = sale.items.count()
        print(f"✓ Final sale verification - Sale ID: {sale.id}, Items: {items_count}, Order: {sale.order}")
        
        
        if sale.order:
            print(f"✓ Sale {sale.id} is properly linked to Order {sale.order.id}")
            
            
            converted_orders = Sales.objects.filter(
                source='order', 
                order__isnull=False
            ).values_list('order_id', flat=True)
            print(f"✓ All converted order IDs: {list(converted_orders)}")
        else:
            print(f"⚠ Sale {sale.id} has no order relationship")
        
        return sale

    def update(self, instance, validated_data):
        item_details_data = validated_data.pop('item_details', [])
        customer_name = validated_data.pop('customer_name', None)
        request = self.context['request']
        user = request.user

        
        if instance.source == 'manual':
            for old_item in instance.items.all():
                old_item.fish.quantity += old_item.quantity
                old_item.fish.save()

        instance.items.all().delete()

        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()

        
        if customer_name:
            try:
                field_names = [field.name for field in Sales._meta.get_fields()]
                if 'customer_name' in field_names:
                    instance.customer_name = customer_name
                    instance.save()
            except Exception as e:
                print(f"Could not set customer_name in update: {e}")

        for item_data in item_details_data:
            fish_name = item_data.get('fish_name')
            quantity = item_data.get('quantity')
            price_per_unit = item_data.get('price_per_unit')
            
            
            try:
                fish = FishStock.objects.get(name__iexact=fish_name.strip(),user=request.user)
            except FishStock.DoesNotExist:
                if instance.source == 'order':
                    raise serializers.ValidationError(f"Fish '{fish_name}' not found in stock.")
                fish = FishStock.objects.create(
                    name=fish_name.strip(),
                    category='Unknown',
                    quantity=0,
                    price='0.00',
                    user=user
                )
            
            if instance.source == 'manual':
                if quantity > fish.quantity:
                    raise serializers.ValidationError(f"Not enough stock for {fish.name}.")
                fish.quantity -= quantity
                fish.save()

            SaleItem.objects.create(
                sale=instance,
                fish=fish,
                quantity=quantity,
                price_per_unit=price_per_unit
            )

        return instance