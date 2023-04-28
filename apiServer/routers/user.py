from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from . import db as db_module

router = APIRouter()

app = FastAPI()
origins = ["*"]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

@router.post("/users", tags=["Users"])
async def create_user(user: db_module.schemas.Users):
  return db_module.crud_user.create_user(user=user)

@router.get("/users", tags=["Users"])
async def get_users():
  users = db_module.crud_user.get_users()
  return users

@router.get("/users/{email}", tags=["Users"])
async def get_user(email: str):
  user = db_module.crud_user.get_user(email=email)
  return user

@router.delete("/users/{email}", tags=["Users"])
async def delete_user(email: str):
  user = db_module.crud_user.delete_user(email=email)
  return user