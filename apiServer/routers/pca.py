from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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


@router.get("/pca", tags=["PCA"])
async def get_pca(symbol: str):
    pca_data = {}
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol=' + \
        symbol + '&apikey=' + api_id_alphavantage
    response = requests.request("GET", url, headers=headers, data=payload)
    main_data = response.json()

    corr_data = {}
    array_open = []
    array_high = []
    array_low = []
    array_close = []
    array_volume = []

    for item in main_data['Weekly Time Series']:
        array_open.append(main_data['Weekly Time Series'][item]['1. open'])
        array_high.append(main_data['Weekly Time Series'][item]['2. high'])
        array_low.append(main_data['Weekly Time Series'][item]['3. low'])
        array_close.append(main_data['Weekly Time Series'][item]['4. close'])
        array_volume.append(main_data['Weekly Time Series'][item]['5. volume'])

    corr_data['open'] = array_open
    corr_data['high'] = array_high
    corr_data['low'] = array_low
    corr_data['close'] = array_close
    corr_data['volume'] = array_volume

    df = pd.DataFrame.from_dict(corr_data)
    text_corr = df.corr().to_json(orient="table")
    json_corr = json.loads(text_corr)

    df = pd.DataFrame.from_dict(main_data)
    estandarizar = StandardScaler()
    m_estandarizada = estandarizar.fit_transform(main_data)
    print(m_estandarizada)
    df = pd.DataFrame.from_dict(m_estandarizada, columns=main_data.columns)
    text_matriz = df.corr().to_json(orient="table")
    json_matriz = json.loads(text_matriz)

    pca_data["main_data"] = main_data
    pca_data["corr_data"] = json_corr
    pca_data["matriz_data"] = json_matriz

    return pca_data
