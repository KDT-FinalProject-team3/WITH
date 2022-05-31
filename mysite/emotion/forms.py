from django import forms
from django.forms import Form
from .models import Document


class DocumentForm(forms.ModelForm):
    # upload = forms.FileField(
    #     label='첨부 파일',
    #     help_text='max. 42 megabytes'
    # )

    class Meta:
        model = Document
        fields = ('txt_file',)

    txt_file = forms.FileField(widget=forms.ClearableFileInput(attrs={'class': 'custom-file-input'}))

