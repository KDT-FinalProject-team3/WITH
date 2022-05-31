from django.urls import path, include
from . import views

app_name = 'emotion'

urlpatterns = [
    path('predict/', views.predict, name='predict'),
    path('', views.index, name='index'),
]
