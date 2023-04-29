from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import routers

description = """
This API developed with the help of Flask will help us connect to Yahoo Finance 🚀

## Base
We will be able to know if the API is in operation or is stopped through various methods. If so, please contact alanfmorag@gmail.com 📧

## Users
With the following routes you can:  
☑️ Insert data from **user**.  
☑️ Get data from **user**.  
☑️ Get a single data from **user**.  
☑️ Delete a single data from **user**.  
☑️ Update **user companies**.  
☑️ Get a **user** by email and password. 
"""

openapi_tags = [
  {
    "name": "Base",
    "description": "Routes to know if the API is active"
  },
  {
    "name": "Users",
    "description": "Routes to get data from Users"
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
app.include_router(routers.user.router)
app.include_router(routers.finance.router)
# app.include_router(routers.amazon.router)
# app.include_router(routers.apple.router)
# app.include_router(routers.tesla.router)