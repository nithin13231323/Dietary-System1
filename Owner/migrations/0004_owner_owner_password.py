# Generated by Django 3.0.4 on 2021-01-22 08:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Owner', '0003_owner_isowner'),
    ]

    operations = [
        migrations.AddField(
            model_name='owner',
            name='Owner_password',
            field=models.CharField(default=0, max_length=32),
            preserve_default=False,
        ),
    ]