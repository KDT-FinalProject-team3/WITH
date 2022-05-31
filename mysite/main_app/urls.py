from django.urls import path, include
from . import views

app_name = 'index'

urlpatterns = [
    path('', views.index, name='index'),

    #path('user/', views.user, name='user'),
    #path('table/', views.table, name='table'),
    #path('typography/', views.typography, name='typography'),
    #path('icons/', views.icons, name='icons'),
    #path('maps/', views.maps, name='maps'),
    #path('notifications/', views.notifications, name='notifications'),
]
