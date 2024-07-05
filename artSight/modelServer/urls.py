from django.urls import path
from . import views

urlpatterns = [
    path('process/', views.process_data, name='process_data'),
    path('art_2d/', views.get_art_2d, name='get_art_2d'),
]
