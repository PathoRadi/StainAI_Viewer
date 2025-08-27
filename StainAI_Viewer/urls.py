from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import include, path

# Define the URL patterns for the project
urlpatterns = [
    # Admin site URL
    path('admin/', admin.site.urls),
    # Include URLs from the 'myapp' application at the root URL
    path('', include('myapp.urls')),
]

# Serve media files through Django during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)