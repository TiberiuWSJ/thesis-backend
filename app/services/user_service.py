from fastapi import HTTPException, status
from sqlmodel import Session
from app.repositories.user_repo import (
    get_user_by_username,
    get_user_by_email,
    create_user as repo_create_user,
    get_user as repo_get_user,
)
from app.models.user import UserCreate, UserRead
from app.core.security import hash_password, verify_password

def register_user(db: Session, user_in: UserCreate) -> UserRead:
    if get_user_by_username(db, user_in.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )
    if get_user_by_email(db, user_in.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    hashed = hash_password(user_in.password)

    user = repo_create_user(db, user_in, hashed)

    return UserRead.from_orm(user)

def authenticate_user(db: Session, username: str, password: str) -> UserRead:
    user = get_user_by_username(db, username)
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return UserRead.from_orm(user)

def get_user(db: Session, user_id: int) -> UserRead:
    user = repo_get_user(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return UserRead.from_orm(user)
