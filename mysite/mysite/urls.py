"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.urls import include


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('main_app.dashboard.urls')),
    path('user/', include('main_app.user.urls')),
    path('icons/', include('main_app.icons.urls')),
    path('table/', include('main_app.table.urls')),
    path('notifications/', include('main_app.notifications.urls')),
    path('typography/', include('main_app.typography.urls')),
    path('emotion/', include('emotion.urls'), name='emotion'),
    path('maps/', include('main_app.maps.urls')),
    path('account/', include('account.urls'))
    # path('maps/',include('main_app.maps.urls')),
]
