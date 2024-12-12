
from django.db import models

# Create your models here.
class Customer(models.Model):
    customer_id = models.AutoField
    customer_firstname = models.CharField(max_length=60)
    customer_lastname = models.CharField(max_length=60)
    customer_address = models.CharField(max_length=600)
    customer_email = models.CharField(max_length=100)
    customer_password = models.CharField(max_length=32)
    customer_dob = models.DateField()
    customer_mobileno = models.CharField(max_length=10)
    customer_gender = models.CharField(max_length=15)
    customer_city = models.CharField(max_length=30)
    customer_state = models.CharField(max_length=30)
    customer_country = models.CharField(max_length=30)
    customer_pincode = models.IntegerField()


    def __str__(self):
        return self.customer_email 





class TextFile(models.Model):
    file = models.FileField(upload_to='uploads/')



    def __str__(self):
        return self.title

class Feedback(models.Model):
    sno = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    address = models.CharField(max_length=500)
    email = models.CharField(max_length=50)
    subject = models.TextField()
    timeStamp = models.DateTimeField(auto_now_add=True, blank=True)


    def __str__(self):
        return 'Feedback from:' + self.name + '-' + self.email


