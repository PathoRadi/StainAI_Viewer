from django.urls import path
from . import views

app_name = 'myapp'
urlpatterns = [
    path('',              views.display_image, name='display_image'),
    path('upload_image/', views.upload_image,    name='upload_image'),
    path('detect_image/', views.detect_image,    name='detect_image'),
    path('delete_project/', views.delete_project, name='delete_project'),
    path('reset-media/', views.reset_media, name='reset_media'),
    path('download_project/', views.download_project, name='download_project'),
    path('download_with_rois/', views.download_project_with_rois, name='download_with_rois'),
    path("progress", views.progress, name="progress")
]
