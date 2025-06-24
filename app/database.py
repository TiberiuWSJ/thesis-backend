# app/database.py
import os
from sqlmodel import SQLModel, create_engine, Session
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    echo=True,
)

def init_db() -> None:
    """
    Create all tables that are defined via SQLModel subclasses.
    """
    SQLModel.metadata.create_all(engine)

def get_db():
    with Session(engine) as session:
        yield session
