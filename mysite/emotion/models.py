from django.db import models


class Document(models.Model):
    txt_file = models.FileField('첨부 파일', upload_to='uploads/')


class EmotionResult(models.Model):
    user = models.ForeignKey('Recipient', models.DO_NOTHING)
    fear = models.CharField(max_length=20)
    surprise = models.CharField(max_length=45)
    anger = models.CharField(max_length=45)
    sadness = models.CharField(max_length=45)
    neutrality = models.CharField(max_length=45)
    happiness = models.CharField(max_length=45)
    anxiety = models.CharField(max_length=45)
    embarrassed = models.CharField(max_length=45)
    hurt = models.CharField(max_length=45)
    interest = models.CharField(max_length=45)
    boredom = models.CharField(max_length=45)
    date = models.DateTimeField()

    class Meta:
        managed = True
        db_table = 'emotion_result'


class Recipient(models.Model):
    user_id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=16)
    contact = models.IntegerField()
    email = models.CharField(max_length=255, blank=True, null=True)
    address = models.CharField(max_length=45)
    birth = models.DateField()
    status = models.CharField(max_length=8)
    create_time = models.DateTimeField(blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'recipient'


class AuthUser(models.Model):
    id = models.CharField(primary_key=True, max_length=12)
    password = models.CharField(max_length=128)
    last_login = models.DateTimeField(blank=True, null=True)
    is_superuser = models.IntegerField()
    username = models.CharField(max_length=150)
    relationship = models.CharField(max_length=8)
    last_name = models.CharField(max_length=150)
    email = models.CharField(max_length=254)
    is_staff = models.IntegerField()
    is_active = models.IntegerField()
    date_joined = models.DateTimeField()
    recipient_id = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'auth_user'

