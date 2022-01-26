from django.urls import path
from .views import ImageUploadView, GetImagesView


urlpatterns = [
    path('upload', ImageUploadView.as_view()),
    path('fetch-images', GetImagesView.as_view()),
]