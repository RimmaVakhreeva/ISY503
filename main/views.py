from django.core.handlers.wsgi import WSGIRequest
from django.shortcuts import render

# Create your views here.
def index(request: WSGIRequest):
    return render(request, 'main/index.html')