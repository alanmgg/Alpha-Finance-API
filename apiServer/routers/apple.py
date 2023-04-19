from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from urllib.request import urlopen
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

@router.get("/eda-get-apple", tags=["Apple"])
async def get_data_apple():
  url = 'https://raw.githubusercontent.com/alanmgg/YFinance-Docs/main/Docs/data_apple.json'
  response = urlopen(url)
  data_json = json.loads(response.read())

  return data_json

@router.get("/eda-tail-apple", tags=["Apple"])
async def tail_data_apple():
  url = 'https://raw.githubusercontent.com/alanmgg/YFinance-Docs/main/Docs/data_apple_tail.json'
  response = urlopen(url)
  data_json = json.loads(response.read())

  return data_json

@router.get("/eda-description-apple", tags=["Apple"])
async def description_data_apple():
  url = 'https://raw.githubusercontent.com/alanmgg/YFinance-Docs/main/Docs/data_apple_description.json'
  response = urlopen(url)
  data_json = json.loads(response.read())

  return data_json

@router.get("/eda-describe-apple", tags=["Apple"])
async def describe_data_apple():
  url = 'https://raw.githubusercontent.com/alanmgg/YFinance-Docs/main/Docs/data_apple_describe.json'
  response = urlopen(url)
  data_json = json.loads(response.read())

  return data_json

@router.get("/eda-corr-apple", tags=["Apple"])
async def corr_data_apple():
  url = 'https://raw.githubusercontent.com/alanmgg/YFinance-Docs/main/Docs/data_apple_corr.json'
  response = urlopen(url)
  data_json = json.loads(response.read())

  return data_json