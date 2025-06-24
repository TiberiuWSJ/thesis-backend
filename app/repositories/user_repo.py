from typing import Optional
from sqlmodel import Session, select
from app.models.user import User, UserCreate

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    stmt = select(User).where(User.username == username)
    return db.exec(stmt).first()

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    stmt = select(User).where(User.email == email)
    return db.exec(stmt).first()

def get_user(db: Session, user_id: int) -> Optional[User]:
    return db.get(User, user_id)

def create_user(db: Session, user: UserCreate, hashed_password: str) -> User:
    db_user = User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
