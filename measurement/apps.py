import sys
from django.apps import AppConfig


class MeasurementConfig(AppConfig):

    default_auto_field = "django.db.models.BigAutoField"
    name = "measurement"

    def ready(self):

        if any(cmd in sys.argv for cmd in [
            "runserver",
            "gunicorn",
            "uwsgi"
        ]):

            from .modules.body_measurement import initialize_models

            initialize_models()