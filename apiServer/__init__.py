from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import routers

description = """
This API developed with the help of Flask will help us connect to Yahoo Finance 🚀

## Base
We will be able to know if the API is in operation or is stopped through various methods. If so, please contact alanfmorag@gmail.com 📧
"""

openapi_tags = [
  {
    "name": "Base",
    "description": "Routes to know if the API is active"
  }
]

app = FastAPI(
  title="Yahoo Finance API",
  description=description,
  version="1.0.1",
  openapi_tags=openapi_tags,
  contact={
    "name": "Alan Francisco Mora",
    "url": "https://alanfmorag.vercel.app/",
    "email": "alanfmorag@gmail.com",
  },
  license_info={
    "name": "MIT License",
    "url": "https://raw.githubusercontent.com/alanmgg/YFinance-API/main/LICENSE",
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