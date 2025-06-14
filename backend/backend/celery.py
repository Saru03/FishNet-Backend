from __future__ import absolute_import, unicode_literals
import os
from celery import Celery
import django
from django.conf import settings
from celery.schedules import crontab

# set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')

app = Celery('backend')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Load task modules from all registered Django app configs.
app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print('Request: {0!r}'.format(self.request))

# Scheduler configuration
app.conf.beat_schedule = {
    'fetch-modis-data-daily': {
        'task': 'fishing_insights.tasks.fetch_modis_data',
        'schedule': crontab(hour=0, minute=0),  # Runs daily at midnight UTC
        'args': (),
    },
}

app.conf.timezone = 'UTC'