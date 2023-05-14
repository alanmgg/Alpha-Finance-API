from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import routers

description = """
Esta API desarrollada con la ayuda de FastAPI nos ayudará a conectarnos a Alpha Vantage 🚀

## Base
Podremos saber si la API está en funcionamiento o está detenida a través de varios métodos. Si es así, póngase en contacto con alanfmorag@gmail.com 📧

## Users
Con las siguientes rutas podrás:  
☑️ Insertar datos de un **usuario**.  
☑️ Obtener datos de todos los **usuarios**.  
☑️ Obtener datos de un solo **usuario**.  
☑️ Eliminar datos de un **usuario**.  
☑️ Actualizar las **compañias de un usuario**.  
☑️ Obtener el **usuario** por su correo y contraseña. 

## Finance
Con las siguientes rutas podrás:  
☑️ Obtener los datos semanales de la **empresa**.  
☑️ Obtener los datos en un intervalo de tiempo de la **empresa**.  
☑️ Obtener las primeras 25 **empresas**.  
☑️ Obtener información de una **empresa**.  
☑️ Obtener una descripción general de la **empresa**.  

## EDA
Con las siguientes rutas podrás:  
☑️ Obtener los **datos principales**.  
☑️ Obtener los **primeros 10 datos**.  
☑️ **Descripción y nulos** de los datos.  
☑️ Describir los datos con ayuda de **pandas**.  
☑️ Obtener los datos **correlacionales**.  
"""

openapi_tags = [
  {
    "name": "Base",
    "description": "Rutas para saber si la API está activa"
  },
  {
    "name": "Users",
    "description": "Rutas para obtener datos de los usuarios"
  },
  {
    "name": "Finance",
    "description": "Rutas que traen datos de la API de Alpha Vantage"
  },
  {
    "name": "EDA",
    "description": "Rutas que traen datos para realizar el proceso EDA"
  }
]

app = FastAPI(
  title="Alpha Finance API",
  description=description,
  version="1.0.3",
  openapi_tags=openapi_tags,
  contact={
    "name": "Alan Francisco Mora",
    "url": "https://www.alanfmorag.tech/",
    "email": "alanfmorag@gmail.com",
  },
  license_info={
    "name": "MIT License",
    "url": "https://raw.githubusercontent.com/alanmgg/Alpha-Finance-API/main/LICENSE",
  }
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

app.include_router(routers.base.router)
app.include_router(routers.user.router)
app.include_router(routers.finance.router)
app.include_router(routers.eda.router)