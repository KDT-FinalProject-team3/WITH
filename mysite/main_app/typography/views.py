from django.shortcuts import render
from django.http import HttpResponse


def index(request):
  #코드 구현
  return render(request, "typography/typography.html")

