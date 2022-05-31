from django.shortcuts import render
from django.http import HttpResponse

def index(request):
  #코드 구현
  return render(request, "dashboard.html")

def user(request): #user 페이지
    return render(request, 'user.html')

def table(request): #table 페이지
    return render(request, 'table.html')

def typography(request): #typography 페이지
    return render(request, 'typography.html')

def icons(request): #icons 페이지
    return render(request, 'icons.html')

def map(request): #map 페이지
    return render(request, 'map.html')

def notifications(request): #notifications 페이지
    return render(request, 'notifications.html')
