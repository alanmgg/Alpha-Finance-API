from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

import requests
import pandas as pd
import json

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

@router.get("/eda-main-data", tags=["EDA"])
async def get_eda_main_data(symbol: str):
  url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=' + symbol + '&apikey=' + api_id_alphavantage
  response = requests.request("GET", url, headers=headers, data=payload)

  return response.json()


@router.get("/eda-tail-data", tags=["EDA"])
async def get_eda_tail_data(symbol: str):
  url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=' + symbol + '&apikey=' + api_id_alphavantage
  response = requests.request("GET", url, headers=headers, data=payload)
  tail = response.json()

  object_tail = {}

  for idx, item in enumerate(tail['Weekly Time Series']):
    if idx == 10:
        break
    object_tail[item] = tail['Weekly Time Series'][item]

  return object_tail


@router.get("/eda-description-data", tags=["EDA"])
async def get_eda_description_data(symbol: str):
  url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=' + symbol + '&apikey=' + api_id_alphavantage
  response = requests.request("GET", url, headers=headers, data=payload)
  description = response.json()

  data = {}
  data['description'] = []
  data['nulls'] = []

  data['description'].append(
    {
      'shape': [len(description['Weekly Time Series']), 5],
      'open': 'float64',
      'high': 'float64',
      'low': 'float64',
      'close': 'float64',
      'volume': 'int64',
      'dtype': 'object'
    }
  )

  data['nulls'].append(
    {
      'open': 0,
      'high': 0,
      'low': 0,
      'close': 0,
      'volume': 0,
      'dtype': 'int64'
    }
  )

  return data


@router.get("/eda-describe-data", tags=["EDA"])
async def get_eda_describe_data(symbol: str):
  url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=' + symbol + '&apikey=' + api_id_alphavantage
  response = requests.request("GET", url, headers=headers, data=payload)
  describe = response.json()

  data = {}
  array_date = []
  array_open = []
  array_high = []
  array_low = []
  array_close = []
  array_volume = []

  for item in describe['Weekly Time Series']:
    array_date.append(item)
    array_open.append(describe['Weekly Time Series'][item]['1. open'])
    array_high.append(describe['Weekly Time Series'][item]['2. high'])
    array_low.append(describe['Weekly Time Series'][item]['3. low'])
    array_close.append(describe['Weekly Time Series'][item]['4. close'])
    array_volume.append(describe['Weekly Time Series'][item]['5. volume'])

  data['date'] = array_date
  data['open'] = array_open
  data['high'] = array_high
  data['low'] = array_low
  data['close'] = array_close
  data['volume'] = array_volume

  df = pd.DataFrame.from_dict(data)
  text_describe = df.describe().to_json(orient="table")
  json_eda_describe = json.loads(text_describe)

  return json_eda_describe


@router.get("/eda-corr-data", tags=["EDA"])
async def get_eda_corr_data(symbol: str):
  url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=' + symbol + '&apikey=' + api_id_alphavantage
  response = requests.request("GET", url, headers=headers, data=payload)
  describe = response.json()

  data = {}
  array_open = []
  array_high = []
  array_low = []
  array_close = []
  array_volume = []

  for item in describe['Weekly Time Series']:
    array_open.append(describe['Weekly Time Series'][item]['1. open'])
    array_high.append(describe['Weekly Time Series'][item]['2. high'])
    array_low.append(describe['Weekly Time Series'][item]['3. low'])
    array_close.append(describe['Weekly Time Series'][item]['4. close'])
    array_volume.append(describe['Weekly Time Series'][item]['5. volume'])

  data['open'] = array_open
  data['high'] = array_high
  data['low'] = array_low
  data['close'] = array_close
  data['volume'] = array_volume

  df = pd.DataFrame.from_dict(data)
  text_corr = df.corr().to_json(orient="table")
  json_eda_corr = json.loads(text_corr)

  return json_eda_corr