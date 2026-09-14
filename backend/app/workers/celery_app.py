from celery import Celery
from app.core.config import get_settings

settings = get_settings()
celery_app = Celery("adaptive_ages", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(task_serializer="json", result_serializer="json", accept_content=["json"], task_track_started=True, worker_prefetch_multiplier=1)
