from django.contrib import admin

# Register your models here.
from .models import Customer
from .models import Feedback

admin.site.register(Customer)
admin.site.register(Feedback)
