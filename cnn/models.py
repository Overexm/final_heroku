from django.db import models



class ImageFind (models.Model):
    image = models.TextField(null=False, blank=False)
    classifier = models.TextField(null=True, blank=True)


    def __str__(self):
        return  self.image
