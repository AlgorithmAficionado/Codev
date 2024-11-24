from rest_framework import routers
from django.urls import path, include
from . import views

router = routers.DefaultRouter()
router.register(r'chat', views.ChatsViewSet, basename="chats")

urlpatterns = [
    path('api/', include(router.urls)),  
]