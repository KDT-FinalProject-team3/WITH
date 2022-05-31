from django.urls import path, include
from . import views

APP_NAME = 'dashboard'

urlpatterns = [
    path('', views.index, name='index'),
    path('dashboard/', views.index, name='index'),
]
