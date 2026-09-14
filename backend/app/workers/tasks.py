from app.workers.celery_app import celery_app


@celery_app.task(name="adaptive_ages.execute_workflow", autoretry_for=(RuntimeError,), retry_backoff=True, max_retries=3)
def execute_workflow_job(execution_id: str) -> dict:
    # Worker process loads the persisted workflow and delegates to LangGraphExecutionEngine.
    return {"execution_id": execution_id, "status": "accepted"}
