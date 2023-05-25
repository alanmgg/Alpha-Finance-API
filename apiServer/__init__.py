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
☑️ Obtener las primeras **500 empresas**.   

## Algorithms
Con las siguientes rutas podrás:  
☑️ Obtener el algoritmo de **Análisis Exploratorio de Datos (EDA)**.  
☑️ Obtener el algoritmo de **Análisis de Componentes Principales (ACP)**.  
☑️ Obtener el algoritmo de **Pronóstico con árboles de decisión**.  
☑️ Obtener el algoritmo de **Pronóstico con bosques aleatorios**.  
☑️ Obtener el algoritmo de **Clasificación con árboles de decisión y bosques aleatorios**.  
☑️ Obtener el algoritmo de **Clustering particional y clasificación**.  
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
    "name": "Algorithms",
    "description": "Rutas hechas para los algoritmos de la APP"
  }
]

app = FastAPI(
  title="Alpha Finance API",
  description=description,
  version="1.0.5",
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
app.include_router(routers.algorithms.router)