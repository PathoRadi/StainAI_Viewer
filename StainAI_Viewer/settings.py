"""
Django settings for StainAI_Viewer project.
"""
import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# ===== Security =====
# keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-!x%ed1b=0sr!#@wln(01vec%rrvl23*jaadib+s0l_-=t89swy'
# don't run with debug turned on in production!
DEBUG = True
# Add your domain names or IP addresses here when deploying
ALLOWED_HOSTS = ['localhost', '127.0.0.1', '.azurewebsites.net']

ALLOWED_HOSTS += ['169.254.130.1', '169.254.130.2', '169.254.130.3', '169.254.130.4']



# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'myapp',
    'whitenoise.runserver_nostatic',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'StainAI_Viewer.urls'


# ===== Templates =====
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'myapp' / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
WSGI_APPLICATION = 'StainAI_Viewer.wsgi.application'



# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}



# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]



# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True



# ===== Static files =====
STATIC_URL = '/static/'
# Source: Since css, js, images are in myapp/static/
STATICFILES_DIRS = [BASE_DIR / 'myapp' / 'static']
# Target: collectstatic will collect files here
STATIC_ROOT = BASE_DIR / 'staticfiles'
# Let Whitenoise automatically compress and add hashed filenames to avoid caching issues
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'



# ===== Media =====
MEDIA_URL  = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'