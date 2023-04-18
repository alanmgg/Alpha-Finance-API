from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

import yfinance as yf
from json import loads
from datetime import date

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

@router.get("/get-spotify", tags=["Spotify"])
async def get_data():
  today = date.today()

  data_spotify = yf.Ticker('SPOT')
  spotify_hist = data_spotify.history(start = '2019-1-1', end = today, interval = '1d')
  json_spotify = spotify_hist.to_json(orient="table")
  result_spotify = loads(json_spotify)
  
  # print (spotify_hist)
  # print (result_spotify['data'][0]['Date'].date())
  return result_spotify