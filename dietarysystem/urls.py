

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('CustomerHome.urls')),
    path('OwnerHome/',include('Owner.urls'))
    

]
