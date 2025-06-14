from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from decimal import Decimal

class FishStock(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='fish_stocks')
    name = models.CharField(max_length=100)
    category = models.CharField(
        max_length=100,
        choices=[('Freshwater', 'Freshwater'), ('Saltwater', 'Saltwater')]
    )
    quantity = models.IntegerField()
    fish_size = models.CharField(
        max_length=50, 
        choices=[
            ('Small', 'Small'),
            ('Medium', 'Medium'),
            ('Large', 'Large'),
        ]
    )
    date_added = models.DateField(null=True, blank=True)
    low_stock_threshold = models.IntegerField(default=10)
    cold_storage = models.CharField(max_length=200)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    description = models.TextField(blank=True)

    def update_quantity(self, amount):
        """Add stock (positive amount) or reduce stock (negative amount)"""
        if self.quantity + amount < 0:
            raise ValidationError(f"Cannot reduce stock below zero. Current: {self.quantity}")
        self.quantity += amount
        self.save()

    def reduce_stock(self, amount):
        """Safely reduce stock with validation"""
        if self.quantity < amount:
            raise ValidationError(f"Insufficient stock. Available: {self.quantity}, Requested: {amount}")
        self.quantity -= amount
        self.save()

    def is_low_stock(self):
        return self.quantity < self.low_stock_threshold

    def __str__(self):
        return f"{self.name} ({self.category}) - {self.quantity} units"
    
    # class Meta:
    #     unique_together = ('user', 'name')
        

class Order(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='orders')
    customer_name = models.CharField(max_length=255, default='Anonymous')
    order_date = models.DateField(auto_now_add=True)
    delivery_date = models.DateField()

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled')
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')

    @property
    def total_amount(self):
        """Calculate total order amount"""
        return sum(item.total_price for item in self.items.all())

    @property
    def total_quantity(self):
        """Calculate total quantity of all items"""
        return sum(item.quantity for item in self.items.all())

    def complete_order(self):
        """Mark order as completed and reduce stock"""
        if self.status == 'completed':
            return
        
        # Validate all items first
        for item in self.items.all():
            item.clean()
        
        # Reduce stock for all items
        for item in self.items.all():
            item.fish.reduce_stock(item.quantity)
        
        self.status = 'completed'
        self.save()

    def __str__(self):
        return f"Order #{self.id} - {self.customer_name} ({self.status})"

class OrderItem(models.Model):
    order = models.ForeignKey(Order, related_name='items', on_delete=models.CASCADE)
    fish = models.ForeignKey(FishStock, on_delete=models.PROTECT)
    quantity = models.IntegerField()
    price_per_unit = models.DecimalField(max_digits=10, decimal_places=2)  # Price at time of order

    @property
    def total_price(self):
        return self.quantity * self.price_per_unit

    def clean(self):
        if self.quantity <= 0:
            raise ValidationError("Quantity must be greater than zero")
        if self.quantity > self.fish.quantity:
            raise ValidationError(f"Not enough stock available for {self.fish.name}. Available: {self.fish.quantity}")
        if self.price_per_unit <= 0:
            raise ValidationError("Price per unit must be greater than zero")

    def save(self, *args, **kwargs):
        # Set default price from fish stock if not provided
        if not self.price_per_unit:
            self.price_per_unit = self.fish.price
        self.clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.quantity} of {self.fish.name} @ ${self.price_per_unit} each"

class Sales(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sales')
    sale_date = models.DateTimeField(auto_now_add=True)
    source = models.CharField(
        max_length=50, 
        choices=[
            ('manual', 'Manual Entry'),
            ('order', 'From Order'),
            ('middle_men', 'Middle Men'),
            ('direct', 'Direct Sale')
        ],
        default='manual'
    )
    order = models.ForeignKey(Order, null=True, blank=True, on_delete=models.SET_NULL, related_name='sales')

    @property
    def total_revenue(self):
        return sum(item.revenue for item in self.items.all())

    @property
    def total_quantity(self):
        return sum(item.quantity for item in self.items.all())

    @property
    def customer_name(self):
        return self.order.customer_name if self.order else "Direct Sale"

    def __str__(self):
        return f"Sale #{self.id} - ${self.total_revenue} ({self.source})"

class SaleItem(models.Model):
    sale = models.ForeignKey(Sales, related_name='items', on_delete=models.CASCADE)
    fish = models.ForeignKey(FishStock, on_delete=models.CASCADE)
    quantity = models.IntegerField()
    price_per_unit = models.DecimalField(max_digits=10, decimal_places=2)  # Changed from FloatField

    @property
    def revenue(self):
        return self.quantity * self.price_per_unit

    @property
    def profit(self):
        """Calculate profit (assuming fish.price is the cost)"""
        cost = self.quantity * self.fish.price
        return self.revenue - cost

    def clean(self):
        if self.quantity <= 0:
            raise ValidationError("Quantity must be greater than zero")
        if self.price_per_unit <= 0:
            raise ValidationError("Price per unit must be greater than zero")

    def save(self, *args, **kwargs):
        self.clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.quantity} of {self.fish.name} @ ${self.price_per_unit} = ${self.revenue}"

class Meta:
    # Add indexes for better performance
    class Meta:
        indexes = [
            models.Index(fields=['user', 'sale_date']),
            models.Index(fields=['source']),
        ]