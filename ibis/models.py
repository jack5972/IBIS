from django.db import models
# from . import settings

class birds_info(models.Model):
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=50)
    size = models.IntegerField()
    colour = models.CharField(max_length=50)
    info = models.TextField(max_length=1000)
    image = models.ImageField(blank=True,upload_to="project_ibis/images")

    def __str__(self):
        return self.name