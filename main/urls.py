from django.urls import path
from . import views as main


urlpatterns = [
    path("", main.index, name='dashboard'),
]
