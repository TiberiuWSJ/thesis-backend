# app/celery_app.py
import os
from celery import Celery
from dotenv import load_dotenv

# Load DATABASE_URL from your project‐root .env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

celery_app = Celery(
    "recon", 
    broker=f"sqla+{DATABASE_URL}",   # use SQLAlchemy transport
    backend=f"db+{DATABASE_URL}",    # result‐backend in the same DB
    include=["app.tasks"],           # where your @task lives
)

# let Celery auto‐create its own tables (kombu, taskmeta…)
celery_app.conf.database_create_tables_at_setup = True