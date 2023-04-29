from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

import requests
from bs4 import BeautifulSoup

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

api_id_alphavantage = '60XFCT31W4YJ0OT6'
api_id_finnhub = 'ch69q3pr01qo6f5d7rngch69q3pr01qo6f5d7ro0'
payload = {}
headers = {}

@router.get("/finance-weekly/{symbol}", tags=["Finance"])
async def get_finance_weekly(symbol: str):
  url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=' + symbol + '&apikey=' + api_id_alphavantage
  response = requests.request("GET", url, headers=headers, data=payload)

  return response.json()

@router.get("/finance-daily/{symbol}/{interval}", tags=["Finance"])
async def get_finance_daily(symbol: str, interval: int):
  url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=' + symbol + '&interval=' + str(interval) + 'min&apikey=' + api_id_alphavantage
  response = requests.request("GET", url, headers=headers, data=payload)

  return response.json()

@router.get("/finance-companies", tags=["Finance"])
async def get_finance_companies():
  url = 'https://es-us.finanzas.yahoo.com/valores-mas-activos/?offset=0&count=50'
  response = requests.request("GET", url, headers=headers, data=payload)
  soup = BeautifulSoup(response.text, 'html.parser')

  table = soup.find('table', {'class': 'W(100%)'})
  rows = table.find_all('tr')[1:]

  companies = []

  for item in rows:
    columns = item.find_all('td')
    if len(columns) > 1:
      symbol = columns[0].text
      name = columns[1].text
      price = columns[2].text
      change = columns[3].text
      change_porcent = columns[4].text
      volume = columns[5].text
      market_capitalization = columns[7].text
      
      companies.append({ 'symbol': symbol, 'name': name, 'price': price, 'change': change,
                        'change_porcent': change_porcent, 'volume': volume, 'market_capitalization': market_capitalization })
   
  return companies

@router.get("/finance-information/{symbol}", tags=["Finance"])
async def get_finance_information(symbol: str):
  url = 'https://www.alphavantage.co/query?function=OVERVIEW&symbol=' + symbol + '&apikey=' + api_id_alphavantage
  response = requests.request("GET", url, headers=headers, data=payload)

  return response.json()

@router.get("/finance-global_quote/{symbol}", tags=["Finance"])
async def get_finance_global_quote(symbol: str):
  url = 'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=' + symbol + '&apikey=' + api_id_alphavantage
  response = requests.request("GET", url, headers=headers, data=payload)

  return response.json()