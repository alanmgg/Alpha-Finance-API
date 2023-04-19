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

@router.get("/eda-get-tesla", tags=["Tesla"])
async def get_data_tesla():
  url = 'https://raw.githubusercontent.com/alanmgg/YFinance-Docs/main/Docs/data_tesla.json'
  response = urlopen(url)
  data_json = json.loads(response.read())

  return data_json

@router.get("/eda-tail-tesla", tags=["Tesla"])
async def tail_data_tesla():
  url = 'https://raw.githubusercontent.com/alanmgg/YFinance-Docs/main/Docs/data_tesla_tail.json'
  response = urlopen(url)
  data_json = json.loads(response.read())

  return data_json

@router.get("/eda-description-tesla", tags=["Tesla"])
async def description_data_tesla():
  url = 'https://raw.githubusercontent.com/alanmgg/YFinance-Docs/main/Docs/data_tesla_description.json'
  response = urlopen(url)
  data_json = json.loads(response.read())

  return data_json

@router.get("/eda-describe-tesla", tags=["Tesla"])
async def describe_data_tesla():
  url = 'https://raw.githubusercontent.com/alanmgg/YFinance-Docs/main/Docs/data_tesla_describe.json'
  response = urlopen(url)
  data_json = json.loads(response.read())

  return data_json

@router.get("/eda-corr-tesla", tags=["Tesla"])
async def corr_data_tesla():
  url = 'https://raw.githubusercontent.com/alanmgg/YFinance-Docs/main/Docs/data_tesla_corr.json'
  response = urlopen(url)
  data_json = json.loads(response.read())

  return data_json