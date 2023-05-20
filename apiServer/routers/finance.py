from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd

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

@router.get("/finance-companies", tags=["Finance"])
async def get_finance_companies():
  companies = []

  symbols = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
  name = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Security']
  founded = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Founded']
  
  for idx, item in enumerate(symbols):
    companies.append({ 'symbol': item, 'name': name[idx], 'founded': founded[idx] })

  return companies