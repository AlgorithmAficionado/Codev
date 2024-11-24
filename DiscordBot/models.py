from django.db import models

class Chats(models.Model):
    query = models.TextField()  
    context = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.query
