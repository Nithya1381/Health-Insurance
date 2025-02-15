from django.urls import path

from . import views

app_name = 'main'

urlpatterns = [
    path('', views.chat, name='chat'),
    path('uploads/', views.uploads, name='uploads'),
    path('documents/', views.view_documents, name='view_documents'),
    path('convert-speech/', views.convert_speech, name='convert_speech'),
    path('delete-document/', views.delete_document, name='delete_document'),
    path('delete-all-documents/', views.delete_all_documents, name='delete_all_documents'),
] 