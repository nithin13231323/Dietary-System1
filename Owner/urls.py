from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from CustomerHome import views as cust_views


urlpatterns = [
    path('', views.index, name="Owner"),
    path('signin/',cust_views.signin, name="SignIn"),
    path('Logout/',cust_views.Logout, name="Logout"),
    path('Profile/',views.Profile, name="Profile"),
   



    path('AllCustomers/',views.AllCustomers, name="AllCustomers"),


    path('CustomerProfile/<str:customer_email>/',views.Customer_Profile,name="CustomerProfile"),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL ,document_root=settings.MEDIA_ROOT)