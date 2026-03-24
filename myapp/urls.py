from django.urls import path
from . import views

app_name = 'myapp'
urlpatterns = [
    path('',              views.display_image, name='display_image'),
    path("auth/login-bridge/", views.viewer_login_bridge, name="viewer_login_bridge"),
    path("api/current-user/", views.current_viewer_user, name="current_viewer_user"),
    path("auth/logout-silent/", views.viewer_logout_silent, name="viewer_logout_silent"),
    path('upload_image/', views.upload_image,    name='upload_image'),
    path('detect_image/', views.detect_image,    name='detect_image'),
    path('delete_image/', views.delete_image, name='delete_image'),
    path('rename_image/', views.rename_image, name='rename_image'),
    path('reset-media/', views.reset_media, name='reset_media'),
    path('download_with_rois/', views.download_project_with_rois, name='download_with_rois'),
    path("progress", views.progress, name="progress"),
    path("api/detect_result/", views.detect_result, name="detect_result"),
    path('create_project/', views.create_project, name='create_project'),
    path('list_projects/', views.list_projects, name='list_projects'),
    path('move_image_to_project/', views.move_image_to_project, name='move_image_to_project'),
    path('move_image_to_images/', views.move_image_to_images, name='move_image_to_images'),
    path('get_project_images/', views.get_project_images, name='get_project_images'),
    path('rename_project/', views.rename_project, name='rename_project'),
    path('delete_project/', views.delete_project, name='delete_project'),
    path("api/viewer-state/", views.viewer_state, name="viewer_state"),
]
