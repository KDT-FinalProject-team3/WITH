
from django.shortcuts import render
from django.http import HttpResponse


def index(request):
  #user 페이지
  return render(request, "user/user.html")

