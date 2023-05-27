from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
import os

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

@router.post("/upload-file/{id_user}", tags=["Files"], response_model=db_module.schemas.Status)
async def upload_file(id_user: int, file: UploadFile = File(...)):
  name = file.filename
  route_local = f"./docs/{file.filename}"
  with open(route_local, "wb") as buffer:
    buffer.write(await file.read())
  file = db_module.crud_files.upload_file(id_user=id_user, filename=file.filename, route_local=route_local)
  if file is None:
    raise HTTPException(status_code=404, detail="Archivo no guardado ...")
  os.remove(route_local)
  return db_module.schemas.Status(message=f"Archivo {name} guardado correctamente")

@router.get("/get-files/{id_user}", tags=["Files"])
async def get_files(id_user: int):
  file = db_module.crud_files.get_files(id_user=id_user)
  if file is None:
    raise HTTPException(status_code=404, detail="Archivos no encontrados ...")
  return file

@router.get("/download-file/{id_user}/{file_name}", tags=["Files"])
async def download_file(id_user: int, file_name: str):
  file = db_module.crud_files.download_file(id_user=id_user, file_name=file_name)
  if file is None:
    raise HTTPException(status_code=404, detail="Archivo no encontrados ...")
  return db_module.schemas.Status(message=f"Archivo {file_name} descargado correctamente")