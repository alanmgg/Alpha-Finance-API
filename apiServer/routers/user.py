from fastapi import FastAPI, APIRouter, HTTPException
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

@router.get("/users/{id_user}", tags=["Users"])
async def get_user(id_user: int):
  user = db_module.crud_user.get_user(id_user=id_user)
  if user is None:
    raise HTTPException(status_code=404, detail="Usuario no encontrado ...")
  return user

@router.post("/users/{email}/{password}", tags=["Users"])
async def get_user_by_email_and_password(email: str, password: str):
  email = email.replace('%40', '@')
  user = db_module.crud_user.get_user_by_email(email=email)
  if user is None:
    raise HTTPException(status_code=404, detail="Usuario no encontrado ...")
  elif user['password'] != password:
    raise HTTPException(status_code=404, detail="Correo o contraseña mal escritos ...")
  return user

@router.delete("/users/{id_user}", tags=["Users"], response_model=db_module.schemas.Status)
async def delete_user(id_user: int):
  user_name = db_module.crud_user.delete_user(id_user=id_user)
  if user_name is None:
    raise HTTPException(status_code=404, detail="Usuario no encontrado ...")
  return db_module.schemas.Status(message=f"Usuario {user_name} eliminado")

@router.put("/users-companies/{id_user}", tags=["Users"], response_model=db_module.schemas.Status)
async def update_user(id_user: int, companies: db_module.schemas.UsersCompanies):
  user = db_module.crud_user.get_user(id_user=id_user)
  if user is None:
    raise HTTPException(status_code=404, detail="Usuario no encontrado ...")
  user_name = db_module.crud_user.update_user(id_user, companies)
  return db_module.schemas.Status(message=f"Usuario {user_name} actualizado")