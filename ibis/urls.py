from django.urls import path
import ibis.views as views

urlpatterns = [
    path('', views.index,name='index'),
    path('getData', views.getData,name='getData'),
    path('show', views.show,name='show'),
    path('description', views.description,name='description'),
    path('redirect', views.redirect,name='redirect'),
    path('compression', views.compression,name='compression'),
]
