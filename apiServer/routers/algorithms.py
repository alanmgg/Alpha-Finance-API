from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime

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

@router.get("/eda", tags=["Algorithms"])
async def get_eda(symbol: str):
  eda_data_finish = {}

  # Configuramos la fecha
  date_now = datetime.now().date()
  format_date = date_now.strftime('%Y-%m-%d')

  # Obtenemos los datos principales
  main_data = yf.Ticker(symbol)
  data_hist = main_data.history(start = '2019-1-1', end = format_date, interval = '1d')
  df = pd.DataFrame.from_dict(data_hist)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  eda_data_finish['main_data'] = json_data['data']

  # Paso 1: Descripción de la estructura de datos
  eda_data_finish['shape'] = data_hist.shape
  eda_data_finish['d_types'] = {
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'volume': 'int64',
    'dividends': 'float64',
    'stock splits': 'float64',
    'dtype': 'object'
  }

  # Paso 2: Identificación de datos faltantes
  eda_data_finish['is_null'] = {
    'open': 0,
    'high': 0,
    'low': 0,
    'close': 0,
    'volume': 0,
    'dividends': 0,
    'stock splits': 0,
    'dtype': 'int64'
  }

  # Paso 3: Detección de valores atípicos
  df = pd.DataFrame.from_dict(data_hist)
  text_data = df.describe().to_json(orient="table")
  json_data = json.loads(text_data)
  eda_data_finish['describe'] = json_data['data']

  # Paso 4: Identificación de relaciones entre pares variables
  df = pd.DataFrame.from_dict(data_hist)
  text_data = df.corr().to_json(orient="table")
  json_data = json.loads(text_data)
  eda_data_finish['corr'] = json_data['data']

  return eda_data_finish


@router.get("/acp", tags=["Algorithms"])
async def get_acp(symbol: str):
  acp_data_finish = {}

  # Configuramos la fecha
  date_now = datetime.now().date()
  format_date = date_now.strftime('%Y-%m-%d')

  # Obtenemos los datos principales
  main_data = yf.Ticker(symbol)
  data_hist = main_data.history(start = '2019-1-1', end = format_date, interval = '1d')
  df = pd.DataFrame.from_dict(data_hist)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  acp_data_finish['main_data'] = json_data['data']

  # Paso 1: Hay evidencia de variables posiblemente correlacionadas
  corr_data = data_hist.corr(method='pearson')
  df = pd.DataFrame.from_dict(corr_data)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  acp_data_finish['corr_data'] = json_data['data']

  # Paso 2: Se hace una estandarización de los datos
  estandarizar = StandardScaler()
  m_estandarizada = estandarizar.fit_transform(data_hist)
  df = pd.DataFrame(m_estandarizada, columns=data_hist.columns)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  acp_data_finish['m_estandarizada'] = json_data['data']

  # Pasos 3 y 4: Se calcula la matriz de covarianzas o correlaciones, y se 
  # calculan los componentes (eigen-vectores) y la varianza (eigen-valores)
  pca = PCA(n_components=7)
  pca.fit(m_estandarizada)
  df = pd.DataFrame(pca.components_)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  acp_data_finish['pca_components'] = json_data['data']

  # Paso 5: Se decide el número de componentes principales
  varianza = pca.explained_variance_ratio_
  df = pd.DataFrame(varianza)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  acp_data_finish['varianza'] = json_data['data']
  acp_data_finish['num_varianza'] = sum(varianza[0:2])

  # Paso 6: Se examina la proporción de relevancias –cargas–
  df = pd.DataFrame(abs(pca.components_))
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  acp_data_finish['cargas'] = json_data['data']

  cargas_components = pd.DataFrame(abs(pca.components_), columns=data_hist.columns)
  text_data = cargas_components.to_json(orient="table")
  json_data = json.loads(text_data)
  acp_data_finish['cargas_components'] = json_data['data']

  data_hist_acp = data_hist.drop(columns=['Volume', 'Dividends', 'Stock Splits'])
  text_data = data_hist_acp.to_json(orient="table")
  json_data = json.loads(text_data)
  acp_data_finish['data_main_acp'] = json_data['data']

  return acp_data_finish


@router.get("/forecast-ad", tags=["Algorithms"])
async def get_forecast_ad(symbol: str):
  ad_data_finish = {}

  # Configuramos la fecha
  date_now = datetime.now().date()
  format_date = date_now.strftime('%Y-%m-%d')

  # Obtenemos los datos principales
  main_data = yf.Ticker(symbol)
  data_hist = main_data.history(start = '2019-1-1', end = format_date, interval = '1d')
  df = pd.DataFrame.from_dict(data_hist)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_data_finish['main_data'] = json_data['data']

  # Descripción de la estructura de los datos
  ad_data_finish['is_null'] = {
    'open': 0,
    'high': 0,
    'low': 0,
    'close': 0,
    'volume': 0,
    'dividends': 0,
    'stock splits': 0,
    'dtype': 'int64'
  }

  df = pd.DataFrame.from_dict(data_hist)
  text_data = df.describe().to_json(orient="table")
  json_data = json.loads(text_data)
  ad_data_finish['describe'] = json_data['data']

  m_datos = data_hist.drop(columns = ['Volume', 'Dividends', 'Stock Splits'])
  df = pd.DataFrame.from_dict(m_datos)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_data_finish['main_data_drop'] = json_data['data']

  # Aplicación del algoritmo
  x = np.array(m_datos[['Open',
                        'High',
                        'Low']])
  df = pd.DataFrame.from_dict(x)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_data_finish['x'] = json_data['data']

  y = np.array(m_datos[['Close']])
  df = pd.DataFrame.from_dict(y)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_data_finish['y'] = json_data['data']

  # Se hace la división de los datos
  x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 0, 
                                                                    shuffle = True)
  df = pd.DataFrame.from_dict(x_test)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_data_finish['x_test'] = json_data['data']

  # Se entrena el modelo
  pronostico_ad = DecisionTreeRegressor(random_state=0)
  pronostico_ad.fit(x_train, y_train)
  # Se genera el pronóstico
  y_pronostico = pronostico_ad.predict(x_test)
  df = pd.DataFrame.from_dict(y_pronostico)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_data_finish['y_pronostico'] = json_data['data']

  ad_data_finish['variables'] = { 'criterion': pronostico_ad.criterion, 
                                  'mae': mean_absolute_error(y_test, y_pronostico),
                                  'mse': mean_squared_error(y_test, y_pronostico),
                                  'rmse': mean_squared_error(y_test, y_pronostico, squared=False),
                                  'score': r2_score(y_test, y_pronostico) }
  
  importancia = pd.DataFrame({'Variable': list(m_datos[['Open', 'High', 'Low']]),
                              'Importancia': pronostico_ad.feature_importances_})
  text_data = importancia.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_data_finish['importancia'] = json_data['data']

  return ad_data_finish