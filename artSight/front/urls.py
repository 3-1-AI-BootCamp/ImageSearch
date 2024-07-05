from django.urls import path
from . import views

urlpatterns = [
    # path('display_data/', views.display_data, name='display_data'),
    path('process/', views.display_data, name='process_data'),
    path('search/', views.requestSearch, name='client_request'),
]
