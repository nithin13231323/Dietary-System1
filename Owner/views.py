from django.shortcuts import render, redirect
from django.http import HttpResponse
from Owner.models import Owner

from CustomerHome.models import Customer

from django.views.decorators.cache import cache_control

from datetime import datetime
from datetime import date
import os
from dietarysystem.settings import MEDIA_ROOT
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
import io, base64

# Create your views here.
@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def index(request):
    if('user_email' not in request.session):
        return redirect('/signin/')
    owner_email = request.session.get('user_email')
    owner = Owner.objects.get(Owner_email=owner_email)
   
    Message="Welcome Aboard!!"

    return render(request,'Owner_index.html',{'Message':Message,'owner':owner})

def Profile(request):
    if('user_email' not in request.session):
        return redirect('/signin/')
    owner_email = request.session.get('user_email')
    owner = Owner.objects.get(Owner_email=owner_email)

    return render(request,'Owner_Profile.html',{'owner':owner})







def AllCustomers(request):
    if('user_email' not in request.session):
        return redirect('/signin/')
    owner_email = request.session.get('user_email')
    owner = Owner.objects.get(Owner_email=owner_email)
    customer = Customer.objects.all()
    
    return render(request,"All_Customers.html",{'customer':customer,'owner':owner})



def Customer_Profile(request,customer_email):
    if('user_email' not in request.session):
        return redirect('/signin/')
    owner_email = request.session.get('user_email')
    owner = Owner.objects.get(Owner_email=owner_email)
    customer = Customer.objects.get(customer_email=customer_email)
   
    return render(request,'Owner_Customer_Profile.html',{'owner':owner})















