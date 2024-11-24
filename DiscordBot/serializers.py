from rest_framework import serializers
from . import models

class ChatSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.Chats
        fields = '__all__'

