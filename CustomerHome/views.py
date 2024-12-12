import sys

from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views.generic import TemplateView
from CustomerHome.models import Customer
from Owner.models import Owner

import random
from django.http import HttpResponseServerError,HttpResponseRedirect
import re

from urllib import request
from django.views.decorators.cache import cache_control

import subprocess
from django.http import JsonResponse
from CustomerHome.models import Feedback

from django.http import FileResponse
from django.core.files.storage import FileSystemStorage
import os
from CustomerHome.forms import FileUploadForm

from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
import signal



from datetime import datetime
from datetime import date

isLogin = False
isLogout = False

# Create your views here.


def index1(request):
    return render(request,'Homenew.html')
def index(request):
    global isLogin
    global isLogout

    if('user_email' in request.session):
        email = request.session.get('user_email')

        result_customer = Customer.objects.filter(customer_email=email)
        result_owner = Owner.objects.filter(Owner_email=email)
       

        if result_customer.exists():
            request.session['user_email'] = email
            isLogin = True
            return redirect('/Home/')
        elif result_owner.exists():
            request.session['user_email'] = email
            isLogin = True
            return redirect('/Owner/')
       
        return redirect('Home')

   
    if('user_email' not in request.session and isLogout):
        isLogin = False
        isLogout = False
        Message = "Successfully Logged Out!!"
    



def signin(request):
    return render(request,'SignIn.html')


def register(request):
    return render(request,'register.html')

def LoginAuthentication(request):
    
    global isLogin
    login_email=request.POST.get('login_email','')
    login_password=request.POST.get('login_password','')
    # customer = Customer.objects.all()

    result_customer = Customer.objects.filter(customer_email=login_email,customer_password=login_password)
    result_owner = Owner.objects.filter(Owner_email=login_email,Owner_password=login_password)
 

    if result_customer.exists():
        request.session['user_email'] = login_email
        isLogin = True
        return redirect('/Home/')
    elif result_owner.exists():
        request.session['user_email'] = login_email
        isLogin = True
        return redirect('/Owner/')
   
    else:
        Message = "Invalid Email or password!!"
        return render(request,'SignIn.html',{'Message':Message})



def RegisterCustomer(request):
    global isLogin
    

    customer_firstname=request.POST.get('customer_firstname','')
    customer_lastname=request.POST.get('customer_lastname','')
    customer_dob=request.POST.get('customer_dob','')
    customer_gender=request.POST.get('customer_gender','')
    customer_mobileno=request.POST.get('customer_mobileno','')
    customer_email=request.POST.get('customer_email','')
    customer_password=request.POST.get('customer_password','')
    confirm_password=request.POST.get('confirm_password','')
    customer_address=request.POST.get('customer_address','')
    customer_city=request.POST.get('customer_city','')
    customer_state=request.POST.get('customer_state','')
    customer_country=request.POST.get('customer_country','')
    customer_pincode=request.POST.get('customer_pincode','')


    if customer_password.isalnum():
       Message = "password should contain a combination of alphabet, number and symbol!!"
       return render(request,'register.html',{'Message':Message})
    
    if (customer_password!=confirm_password):
        Message = "Both the passwords are not same!!!"
        return render(request,'register.html',{'Message':Message})



    result_customer = Customer.objects.filter(customer_email=customer_email)
    result_owner = Owner.objects.filter(Owner_email=customer_email)
   

    if result_customer.exists() or result_owner.exists():
        Message = "This Email address already exist!!"
        return render(request,'register.html',{'Message':Message})
    else:
        customer=Customer(customer_firstname=customer_firstname,customer_lastname=customer_lastname,
        customer_dob=customer_dob,customer_gender=customer_gender,customer_mobileno=customer_mobileno,
        customer_email=customer_email,customer_password=customer_password,customer_address=customer_address,
        customer_city=customer_city,customer_state=customer_state,customer_country=customer_country,
        customer_pincode=customer_pincode)
        
        customer.save()
        request.session['user_email'] = customer_email
        isLogin = True
        return redirect('/Homenew/')


def Logout(request):
    global isLogout
    del request.session['user_email']
    isLogout = True
    Message = "Successfully Logged Out!!"
    return redirect('/')


@cache_control(no_cache=True, must_revalidate=True, no_store=True)
def Home(request):
    if('user_email' not in request.session):
        return redirect('/signin/')
    customer_email = request.session.get('user_email')
    customer = Customer.objects.get(customer_email=customer_email)
    
    Message="Welcome Aboard!!"
    return render(request,'Home.html',{'Message':Message,'customer':customer})

def Profile(request):
    if('user_email' not in request.session):
        return redirect('/signin/')
    customer_email = request.session.get('user_email')
    customer = Customer.objects.get(customer_email=customer_email)
    return render(request,'Profile.html',{'customer':customer})







def about_us(request):
    return HttpResponse('About Us')
    
def contact_us(request):
    return HttpResponse('Contact Us')

def search(request):
    return HttpResponse('search')

def feedback(request):
    return render(request,'feedback.html')
















    





from django.contrib import messages

def feedback(request):
    if request.method == 'POST':
        name = request.POST['name']
        address = request.POST['address']
        email = request.POST['email']
        subject = request.POST['subject']
        print(name,address,email,subject)
        feedback = Feedback(name=name, address=address, email=email, subject=subject)
        feedback.save()
        messages.success(request, 'Thank you for your feedback!')
        return redirect('feedback')
    else:
        return render(request, 'feedback.html')



def feedback_details(request):
    feedback = Feedback.objects.all().order_by('-timeStamp')
    context = {'feedback': feedback}
    return render(request, 'feedback_details.html', context)


def feedback_display(request, pk):
    feedback = Feedback.objects.get(pk=pk)
    context = {'feedback': feedback}
    return render(request, 'feedback_display.html', context)










from .Diet_plans import generate_meal_plan, data, compute_bmr, compute_daily_caloric_intake
from .suggest_alternative import get_recommendations_for_meal

def recommend_view(request):
    if request.method == "POST":
        # Extract user inputs from the form
        gender = request.POST.get('gender')
        weight = float(request.POST.get('weight'))
        height = float(request.POST.get('height'))
        age = int(request.POST.get('age'))
        activity = request.POST.get('activity')
        objective = request.POST.get('objective')

        # Calculate Basal Metabolic Rate (BMR)
        bmr_value = compute_bmr(gender, weight, height, age)
        
        # Calculate daily caloric intake
        daily_caloric_intake_value = compute_daily_caloric_intake(bmr_value, activity, objective)

        # Generate the meal plan
        meal_plan = generate_meal_plan(
            category=gender,
            body_weight=weight,
            body_height=height,
            age=age,
            activity_intensity=activity,
            objective=objective,
            recipes_df=data,
            tolerance=50  # Adjust as needed
        )

        # Get alternative recommendations
        recommendations = {}
        for meal in ['breakfast', 'lunch', 'dinner']:
            meal_name = meal_plan[meal]['Name'] if meal_plan[meal] is not None else None
            if meal_name:
                alternatives = get_recommendations_for_meal(meal_name)
                recommendations[meal] = {
                    'main': meal_name,
                    'alternatives': alternatives[['Name', 'Calories']].to_dict(orient='records') if alternatives is not None else []
                }
            else:
                recommendations[meal] = {'main': 'No recommendation', 'alternatives': []}

        # Pass daily caloric intake and recommendations to the template
        return render(request, 'LoggedinBase.html', {
            'recommendations': recommendations,
            'total_calories': daily_caloric_intake_value  # Pass daily caloric intake to the template
        })

    # Render the form if GET request
    return render(request, 'diet/LoggedinBase.html')


from django.shortcuts import render
from django.http import Http404
import pandas as pd
from urllib.parse import unquote


# Load the dataset
recipes_df = pd.read_csv('/Users/nithinabraham/Downloads/main project/Dietary System copy 2/recipes.csv')
print(recipes_df.columns)
print(recipes_df.head())


# Load the dataset globally
try:
    recipes_df = pd.read_csv('/Users/nithinabraham/Downloads/main project/Dietary System copy 2/recipes.csv')
except FileNotFoundError:
    raise Exception("The recipes.csv file was not found. Check the path.")

def recipe_details(request, food_name):
    try:
        # Decode the URL-encoded food name
        food_name = unquote(food_name)

        # Match the food name in the dataset
        recipe = recipes_df[recipes_df['Name'].str.lower() == food_name.lower()].iloc[0]

        # Get the recipe instructions
        recipe_instructions = recipe['RecipeInstructions']
    except IndexError:
        raise Http404("Recipe not found")
    
    # Pass the instructions to the template
    return render(request, 'recipe_details.html', {'food_name': food_name, 'instructions': recipe_instructions})

