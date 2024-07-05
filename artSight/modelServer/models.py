from django.db import models

# Create your models here.
class Art2D(models.Model):
    
    id = models.AutoField(primary_key=True)
    imageFileName = models.CharField(max_length=255)
    MainCategory = models.CharField(max_length=255)
    SubCategory = models.CharField(max_length=255)
    MiddleCategory = models.CharField(max_length=255)
    Class_kor = models.CharField(max_length=255)
    ArtTitle_kor = models.CharField(max_length=255)
    ArtistName_kor = models.CharField(max_length=255)
    Image = models.BinaryField()

    class Meta:
        db_table = 'art_2d'