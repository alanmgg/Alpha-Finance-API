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

@router.post("/user", tags=["Users"])
async def create_user(user: db_module.schemas.Users):
  return db_module.crud_user.create_user(user=user)