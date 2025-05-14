"""
Django settings for StainAI_Viewer project.

For more info on settings, see:
https://docs.djangoproject.com/en/5.0/topics/settings/
"""

from pathlib import Path

# ─── BASE DIRECTORY ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ─── SECURITY & DEBUG ─────────────────────────────────────────────────────
SECRET_KEY = 'django-insecure-!x%ed1b=0sr!#@wln(01vec%rrvl23*jaadib+s0l_-=t89swy'
DEBUG = True
ALLOWED_HOSTS = ['*']   # In prod, lock this down to your domain(s)

# ─── APPLICATIONS ─────────────────────────────────────────────────────────
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',  # needed to manage static files
    'myapp',                       # your app
]

# ─── MIDDLEWARE ────────────────────────────────────────────────────────────
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # serve static files without extra webserver
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'StainAI_Viewer.urls'

# ─── TEMPLATES ────────────────────────────────────────────────────────────
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],            # add extra template dirs here if you have them
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

# ─── DATABASE ───────────────────────────────────────────────────────────────
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# ─── PASSWORD VALIDATION ───────────────────────────────────────────────────
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',},
]

# ─── INTERNATIONALIZATION ───────────────────────────────────────────────────
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# ─── STATIC FILES (CSS, JS, IMAGES) ────────────────────────────────────────
STATIC_URL = '/static/'                                    # URL prefix for static files
STATIC_ROOT = BASE_DIR / 'staticfiles'                     # where collectstatic will put files for production
STATICFILES_DIRS = [                                       # where Django looks for static files in dev
    BASE_DIR / 'myapp' / 'static',
]

# ─── DEFAULT PRIMARY KEY FIELD TYPE ────────────────────────────────────────
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
