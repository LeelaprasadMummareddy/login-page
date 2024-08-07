from django.contrib import admin
from django.urls import path


from . import views

urlpatterns = [
    path("", views.index, name='index'),
    path("about/", views.about, name="about"),
    path("registration/", views.registration, name="registration"),
    path("login/", views.login, name="login"),
    path("userhome/", views.userhome, name="userhome"),
    path("load/", views.load, name="load"),
    path("view/", views.view, name="view"),
    path("prepro/", views.prepro, name="prepro"),
    path("modules/", views.modules, name="modules"),
    path('prediction/', views.prediction, name = "prediction"),
    path('graph/', views.graph, name = "graph"),

]
