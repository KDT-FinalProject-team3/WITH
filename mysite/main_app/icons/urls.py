from django.urls import path, include
from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('icons', views.index, name='index'),
]
