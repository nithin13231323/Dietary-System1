from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.conf.urls import include

urlpatterns = [
    path('', views.index1, name="Homenew"),
    path('Homenew/', views.index, name="Home"),
    path('Home/', views.Home, name="LoggedinHome"),
    path('signin/',views.signin,name="SignIn"),
    path('Logout/',views.Logout,name="Logout"), 
    path('feedback/', views.feedback, name="feedback"),   
    path('feedback/', views.feedback, name="feedback"), 
    path('feedback_details/', views.feedback_details, name="feedback_details"),
    path('feedback_display/<int:pk>/', views.feedback_display, name='feedback_display'),
    path('home/', views.Home, name='home'),

    path('/recommend_view', views.recommend_view, name='recommend'),
    path('recipe/<str:food_name>/', views.recipe_details, name='recipe_details'),

    
    
    path('register/',views.register,name="Register"),
    path('Profile/',views.Profile,name="Profile"),
    path('about/', views.about_us, name="AboutUs"),
    
    
    path('contact/', views.contact_us, name="ContactUs"),
    path('search/', views.search, name="Search"),
    path('LoginAuthentication/',views.LoginAuthentication,name="LoginAuthentication"),
    path('RegisterCustomer/',views.RegisterCustomer,name="RegisterCustomer"),
    
   
    path('Owner/',include("Owner.urls"))
   
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL ,document_root=settings.MEDIA_ROOT)