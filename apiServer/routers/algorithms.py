from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn import model_selection
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, roc_curve, auc, pairwise_distances_argmin_min
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

from . import db as db_module

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
  df = pd.DataFrame.from_dict(y_test)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_data_finish['y_test'] = json_data['data']

  valores = pd.DataFrame(y_test, y_pronostico)
  text_data = valores.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_data_finish['valores'] = json_data['data']

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


@router.get("/forecast-ba", tags=["Algorithms"])
async def get_forecast_ba(symbol: str):
  ba_data_finish = {}

  # Configuramos la fecha
  date_now = datetime.now().date()
  format_date = date_now.strftime('%Y-%m-%d')

  # Obtenemos los datos principales
  main_data = yf.Ticker(symbol)
  data_hist = main_data.history(start = '2019-1-1', end = format_date, interval = '1d')
  df = pd.DataFrame.from_dict(data_hist)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  ba_data_finish['main_data'] = json_data['data']

  # Descripción de la estructura de los datos
  ba_data_finish['is_null'] = {
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
  ba_data_finish['describe'] = json_data['data']

  m_datos = data_hist.drop(columns = ['Volume', 'Dividends', 'Stock Splits'])
  df = pd.DataFrame.from_dict(m_datos)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  ba_data_finish['main_data_drop'] = json_data['data']

  # Aplicación del algoritmo
  x = np.array(m_datos[['Open',
                        'High',
                        'Low']])
  df = pd.DataFrame.from_dict(x)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  ba_data_finish['x'] = json_data['data']

  y = np.array(m_datos[['Close']])
  df = pd.DataFrame.from_dict(y)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  ba_data_finish['y'] = json_data['data']

  # Se hace la división de los datos
  x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 0, 
                                                                    shuffle = True)
  df = pd.DataFrame.from_dict(x_test)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  ba_data_finish['x_test'] = json_data['data']

  # Se entrena el modelo
  pronostico_ba = RandomForestRegressor(random_state=0)
  pronostico_ba.fit(x_train, y_train)
  # Se genera el pronóstico
  y_pronostico = pronostico_ba.predict(x_test)
  df = pd.DataFrame.from_dict(y_pronostico)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  ba_data_finish['y_pronostico'] = json_data['data']
  df = pd.DataFrame.from_dict(y_test)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  ba_data_finish['y_test'] = json_data['data']

  valores = pd.DataFrame(y_test, y_pronostico)
  text_data = valores.to_json(orient="table")
  json_data = json.loads(text_data)
  ba_data_finish['valores'] = json_data['data']

  ba_data_finish['variables'] = { 'criterion': pronostico_ba.criterion, 
                                  'mae': mean_absolute_error(y_test, y_pronostico),
                                  'mse': mean_squared_error(y_test, y_pronostico),
                                  'rmse': mean_squared_error(y_test, y_pronostico, squared=False),
                                  'score': r2_score(y_test, y_pronostico) }
  
  importancia = pd.DataFrame({'Variable': list(m_datos[['Open', 'High', 'Low']]),
                              'Importancia': pronostico_ba.feature_importances_})
  text_data = importancia.to_json(orient="table")
  json_data = json.loads(text_data)
  ba_data_finish['importancia'] = json_data['data']

  return ba_data_finish


@router.get("/classification-ad-ba", tags=["Algorithms"])
async def get_forecast_ad_ba():
  ad_ba_data_finish = {}

  url = "https://raw.githubusercontent.com/alanmgg/Data-Mining/main/Proyecto/Drug.csv"
  data_ad_ba = pd.read_csv(url)
  text_data = data_ad_ba.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_ba_data_finish['main_data'] = json_data['data']

  # Sexo del paciente
  data_ad_ba.loc[data_ad_ba['Sex'] == 'F', 'Sex'] = 1.0
  data_ad_ba.loc[data_ad_ba['Sex'] == 'M', 'Sex'] = 2.0
  # Presión sanguinea
  data_ad_ba.loc[data_ad_ba['BP'] == 'LOW', 'BP'] = 1.0
  data_ad_ba.loc[data_ad_ba['BP'] == 'NORMAL', 'BP'] = 2.0
  data_ad_ba.loc[data_ad_ba['BP'] == 'HIGH', 'BP'] = 3.0
  # Nivel del colesterol
  data_ad_ba.loc[data_ad_ba['Cholesterol'] == 'NORMAL', 'Cholesterol'] = 1.0
  data_ad_ba.loc[data_ad_ba['Cholesterol'] == 'HIGH', 'Cholesterol'] = 2.0
  # Medicamento que funciono con ese paciente
  data_ad_ba.loc[data_ad_ba['Drug'] == 'drugA', 'Drug'] = 1.0
  data_ad_ba.loc[data_ad_ba['Drug'] == 'drugB', 'Drug'] = 2.0
  data_ad_ba.loc[data_ad_ba['Drug'] == 'drugC', 'Drug'] = 3.0
  data_ad_ba.loc[data_ad_ba['Drug'] == 'drugX', 'Drug'] = 4.0
  data_ad_ba.loc[data_ad_ba['Drug'] == 'drugY', 'Drug'] = 5.0
  # Convertir la columna en tipo float
  data_ad_ba['Sex'] = data_ad_ba['Sex'].astype(float)
  data_ad_ba['BP'] = data_ad_ba['BP'].astype(float)
  data_ad_ba['Cholesterol'] = data_ad_ba['Cholesterol'].astype(float)
  data_ad_ba['Drug'] = data_ad_ba['Drug'].astype(float)
  text_data = data_ad_ba.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_ba_data_finish['main_data_new'] = json_data['data']

  # Descripción de la estructura de los datos
  ad_ba_data_finish['is_null'] = {
    'age': 0,
    'sex': 0,
    'bp': 0,
    'cholesterol': 0,
    'na_to_k': 0,
    'drug': 0,
    'dtype': 'float64(5), int64(1)'
  }

  text_data = data_ad_ba.groupby('Drug').size().to_json(orient="table")
  json_data = json.loads(text_data)
  ad_ba_data_finish['groupby'] = json_data['data']
  
  text_data = data_ad_ba.describe().to_json(orient="table")
  json_data = json.loads(text_data)
  ad_ba_data_finish['describe'] = json_data['data']

  text_data = data_ad_ba.corr().to_json(orient="table")
  json_data = json.loads(text_data)
  ad_ba_data_finish['corr'] = json_data['data']

  # Variables predictoras
  x = np.array(data_ad_ba[['Age',
                            'Sex',
                            'BP',
                            'Cholesterol',
                            'Na_to_K']])
  df = pd.DataFrame(x)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_ba_data_finish['x'] = json_data['data']

  # Variable clase
  y = np.array(data_ad_ba[['Drug']])
  df = pd.DataFrame(y)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_ba_data_finish['y'] = json_data['data']

  x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)

  df = pd.DataFrame.from_dict(x_train)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_ba_data_finish['x_train'] = json_data['data']

  # Se entrena el modelo a partir de los datos de entrada
  clasificacion_ad = DecisionTreeClassifier(random_state=0)
  clasificacion_ad.fit(x_train, y_train)
  # Clasificación final 
  y_clasificacion_ad = clasificacion_ad.predict(x_validation)
  lista_array = y_clasificacion_ad.tolist()
  ad_ba_data_finish['y_clasificacion_ad'] = lista_array

  valores_mod_1 = pd.DataFrame(y_validation, y_clasificacion_ad)
  text_data = valores_mod_1.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_ba_data_finish['valores_mod_1'] = json_data['data']

  ad_ba_data_finish['accuracy_score_ad'] = accuracy_score(y_validation, y_clasificacion_ad)

  # Matriz de clasificación
  modelo_clasificacion_1 = clasificacion_ad.predict(x_validation)
  matriz_clasificacion_1 = pd.crosstab(y_validation.ravel(), 
                                    modelo_clasificacion_1, 
                                    rownames=['Actual'], 
                                    colnames=['Clasificación'])
  text_data = matriz_clasificacion_1.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_ba_data_finish['matriz_clasificacion_1'] = json_data['data']

  ad_ba_data_finish['variables_ad'] = { 'criterion': clasificacion_ad.criterion, 
                                        'exactitud': accuracy_score(y_validation, y_clasificacion_ad) }
  
  importancia_mod_1 = pd.DataFrame({'Variable': list(data_ad_ba[['Age',
                                                              'Sex',
                                                              'BP',
                                                              'Cholesterol',
                                                              'Na_to_K']]),
                                'Importancia': clasificacion_ad.feature_importances_}).sort_values('Importancia', ascending=False)
  text_data = importancia_mod_1.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_ba_data_finish['importancia_mod_1'] = json_data['data']

  clasificacion_ba = RandomForestClassifier(random_state=0)
  clasificacion_ba.fit(x_train, y_train)
  y_clasificacion_ba = clasificacion_ba.predict(x_validation)
  lista_array = y_clasificacion_ba.tolist()
  ad_ba_data_finish['y_clasificacion_ba'] = lista_array

  valores_mod_2 = pd.DataFrame(y_validation, y_clasificacion_ba)
  text_data = valores_mod_2.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_ba_data_finish['valores_mod_2'] = json_data['data']

  ad_ba_data_finish['accuracy_score_ba'] = accuracy_score(y_validation, y_clasificacion_ba)

  # Matriz de clasificación
  modelo_clasificacion_2 = clasificacion_ba.predict(x_validation)
  matriz_clasificacion_2 = pd.crosstab(y_validation.ravel(),
                                      modelo_clasificacion_2,
                                      rownames=['Reales'],
                                      colnames=['Clasificación'])
  text_data = matriz_clasificacion_2.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_ba_data_finish['matriz_clasificacion_2'] = json_data['data']

  ad_ba_data_finish['variables_ba'] = { 'criterion': clasificacion_ba.criterion, 
                                        'exactitud': accuracy_score(y_validation, y_clasificacion_ba) }

  importancia_mod_2 = pd.DataFrame({'Variable': list(data_ad_ba[['Age',
                                                          'Sex',
                                                          'BP',
                                                          'Cholesterol',
                                                          'Na_to_K']]), 
                             'Importancia': clasificacion_ba.feature_importances_}).sort_values('Importancia', ascending=False)
  text_data = importancia_mod_2.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_ba_data_finish['importancia_mod_2'] = json_data['data']

  ad_ba_data_finish['validation'] = { 'ad': accuracy_score(y_validation, y_clasificacion_ad), 
                                      'ba': accuracy_score(y_validation, y_clasificacion_ba) }
  
  # Rendimiento
  y_score = clasificacion_ba.predict_proba(x_validation)
  y_test_bin = label_binarize(y_validation, classes=[1, 
                                                    2, 
                                                    3,
                                                    4,
                                                    5])
  n_classes = y_test_bin.shape[1]
  # Se calcula la curva ROC y el área bajo la curva para cada clase
  fpr = dict()
  tpr = dict()
  response = []
  for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    response.append(auc(fpr[i], tpr[i]))
  ad_ba_data_finish['auc'] = response

  return ad_ba_data_finish


@router.get("/segmentation-classification", tags=["Algorithms"])
async def get_segmentation():
  segmentation_data_finish = {}

  url = "https://raw.githubusercontent.com/alanmgg/Data-Mining/main/Proyecto/CustomerSegmentation.csv"
  data_clientes = pd.read_csv(url)
  text_data = data_clientes.to_json(orient="table")
  json_data = json.loads(text_data)
  segmentation_data_finish['main_data'] = json_data['data']

  # Sexo del cliente
  data_clientes.loc[data_clientes['Gender'] == 'Female', 'Gender'] = 1.0
  data_clientes.loc[data_clientes['Gender'] == 'Male', 'Gender'] = 2.0
  # Estado civil
  data_clientes.loc[data_clientes['Ever_Married'] == 'No', 'Ever_Married'] = 1.0
  data_clientes.loc[data_clientes['Ever_Married'] == 'Yes', 'Ever_Married'] = 2.0
  # Graduado
  data_clientes.loc[data_clientes['Graduated'] == 'No', 'Graduated'] = 1.0
  data_clientes.loc[data_clientes['Graduated'] == 'Yes', 'Graduated'] = 2.0
  # Profesión
  data_clientes.loc[data_clientes['Profession'] == 'Artist', 'Profession'] = 1.0
  data_clientes.loc[data_clientes['Profession'] == 'Doctor', 'Profession'] = 2.0
  data_clientes.loc[data_clientes['Profession'] == 'Engineer', 'Profession'] = 3.0
  data_clientes.loc[data_clientes['Profession'] == 'Entertainment', 'Profession'] = 4.0
  data_clientes.loc[data_clientes['Profession'] == 'Executive', 'Profession'] = 5.0
  data_clientes.loc[data_clientes['Profession'] == 'Healthcare', 'Profession'] = 6.0
  data_clientes.loc[data_clientes['Profession'] == 'Homemaker', 'Profession'] = 7.0
  data_clientes.loc[data_clientes['Profession'] == 'Lawyer', 'Profession'] = 8.0
  data_clientes.loc[data_clientes['Profession'] == 'Marketing', 'Profession'] = 9.0
  # Score
  data_clientes.loc[data_clientes['Spending_Score'] == 'High', 'Spending_Score'] = 1.0
  data_clientes.loc[data_clientes['Spending_Score'] == 'Average', 'Spending_Score'] = 2.0
  data_clientes.loc[data_clientes['Spending_Score'] == 'Low', 'Spending_Score'] = 3.0
  # Medicamento que funciono con ese paciente
  data_clientes.loc[data_clientes['Var_1'] == 'Cat_1', 'Var_1'] = 1.0
  data_clientes.loc[data_clientes['Var_1'] == 'Cat_2', 'Var_1'] = 2.0
  data_clientes.loc[data_clientes['Var_1'] == 'Cat_3', 'Var_1'] = 3.0
  data_clientes.loc[data_clientes['Var_1'] == 'Cat_4', 'Var_1'] = 4.0
  data_clientes.loc[data_clientes['Var_1'] == 'Cat_5', 'Var_1'] = 5.0
  data_clientes.loc[data_clientes['Var_1'] == 'Cat_6', 'Var_1'] = 6.0
  data_clientes.loc[data_clientes['Var_1'] == 'Cat_7', 'Var_1'] = 7.0
  # Reemplazar datos nulos por un 99
  data_clientes.fillna(99, inplace=True)
  # Convertir la columna en tipo float
  data_clientes['Gender'] = data_clientes['Gender'].astype(float)
  data_clientes['Ever_Married'] = data_clientes['Ever_Married'].astype(float)
  data_clientes['Graduated'] = data_clientes['Graduated'].astype(float)
  data_clientes['Profession'] = data_clientes['Profession'].astype(float)
  data_clientes['Spending_Score'] = data_clientes['Spending_Score'].astype(float)
  data_clientes['Var_1'] = data_clientes['Var_1'].astype(float)
  text_data = data_clientes.to_json(orient="table")
  json_data = json.loads(text_data)
  segmentation_data_finish['main_data_new'] = json_data['data']

  # Descripción de la estructura de los datos
  segmentation_data_finish['is_null'] = {
    'ID': 0,
    'Gender': 0,
    'Ever_Married': 0,
    'Age': 0,
    'Graduated': 0,
    'Profession': 0,
    'Work_Experience': 0,
    'Spending_Score': 0,
    'Family_Size': 0,
    'Var_1': 0,
    'dtype': 'float64(5), int64(1)'
  }

  text_data = data_clientes.groupby('Gender').size().to_json(orient="table")
  json_data = json.loads(text_data)
  segmentation_data_finish['groupby'] = json_data['data']
  
  data_clientes.drop(['ID'], axis=1, inplace=True)
  text_data = data_clientes.to_json(orient="table")
  json_data = json.loads(text_data)
  segmentation_data_finish['main_data_new_drop'] = json_data['data']

  text_data = data_clientes.corr().to_json(orient="table")
  json_data = json.loads(text_data)
  segmentation_data_finish['corr'] = json_data['data']

  estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
  m_estandarizada = estandarizar.fit_transform(data_clientes)    # Se calculan la media y desviación y se escalan los datos
  df = pd.DataFrame(m_estandarizada)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  segmentation_data_finish['m_estandarizada'] = json_data['data']

  # Definición de k clusters para K-means
  # Se utiliza random_state para inicializar el generador interno de números aleatorios
  sse = []
  for i in range(2, 10):
      km = KMeans(n_clusters=i, random_state=0)
      km.fit(m_estandarizada)
      sse.append(km.inertia_)
  segmentation_data_finish['sse'] = sse

  # Se crean las etiquetas de los elementos en los clusters
  m_particional = KMeans(n_clusters=4, random_state=0).fit(m_estandarizada)
  m_particional.predict(m_estandarizada)
  data_clientes['ClusterC'] = m_particional.labels_
  text_data = data_clientes.to_json(orient="table")
  json_data = json.loads(text_data)
  segmentation_data_finish['matriz_cluster'] = json_data['data']

  text_data = data_clientes.groupby('ClusterC').size().to_json(orient="table")
  json_data = json.loads(text_data)
  segmentation_data_finish['groupby_cluster'] = json_data['data']

  text_data = data_clientes.groupby('ClusterC').mean().to_json(orient="table")
  json_data = json.loads(text_data)
  segmentation_data_finish['centroides'] = json_data['data']
  
  lista_array = m_estandarizada[:, 0].tolist()
  segmentation_data_finish['scatter_m_est_x'] = lista_array
  lista_array = m_estandarizada[:, 1].tolist()
  segmentation_data_finish['scatter_m_est_y'] = lista_array
  lista_array = m_estandarizada[:, 2].tolist()
  segmentation_data_finish['scatter_m_est_z'] = lista_array
  lista_array = m_particional.cluster_centers_[:, 0].tolist()
  segmentation_data_finish['scatter_m_par_x'] = lista_array
  lista_array = m_particional.cluster_centers_[:, 1].tolist()
  segmentation_data_finish['scatter_m_par_y'] = lista_array
  lista_array = m_particional.cluster_centers_[:, 2].tolist()
  segmentation_data_finish['scatter_m_par_z'] = lista_array

  # Variables predictoras
  x = np.array(data_clientes[['Gender', 
                              'Ever_Married', 
                              'Age', 
                              'Graduated', 
                              'Profession', 
                              'Work_Experience',
                              'Spending_Score',
                              'Family_Size',
                              'Var_1']])
  df = pd.DataFrame(x)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  segmentation_data_finish['x'] = json_data['data']

  # Variable clase
  y = np.array(data_clientes[['ClusterC']])
  df = pd.DataFrame(y)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  segmentation_data_finish['y'] = json_data['data']

  x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)
  df = pd.DataFrame.from_dict(x_train)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  segmentation_data_finish['x_train'] = json_data['data']

  clasificacion_ba = RandomForestClassifier(n_estimators=105,
                                         max_depth=8, 
                                         min_samples_split=4, 
                                         min_samples_leaf=2, 
                                         random_state=1234)
  clasificacion_ba.fit(x_train, y_train)
  # Clasificación final 
  y_clasificacion_ba = clasificacion_ba.predict(x_validation)
  lista_array = y_clasificacion_ba.tolist()
  segmentation_data_finish['y_clasificacion_ba'] = lista_array

  valores_ba = pd.DataFrame(y_validation, y_clasificacion_ba)
  text_data = valores_ba.to_json(orient="table")
  json_data = json.loads(text_data)
  segmentation_data_finish['valores_ba'] = json_data['data']
  
  segmentation_data_finish['accuracy_score'] = accuracy_score(y_validation, y_clasificacion_ba)

  # Matriz de clasificación
  modelo_clasificacion = clasificacion_ba.predict(x_validation)
  matriz_clasificacion = pd.crosstab(y_validation.ravel(),
                                      modelo_clasificacion,
                                      rownames=['Reales'],
                                      colnames=['Clasificación'])
  text_data = matriz_clasificacion.to_json(orient="table")
  json_data = json.loads(text_data)
  segmentation_data_finish['matriz_clasificacion'] = json_data['data']

  segmentation_data_finish['variables'] = { 'criterion': clasificacion_ba.criterion, 
                                            'exactitud': accuracy_score(y_validation, y_clasificacion_ba) }

  importancia = pd.DataFrame({'Variable': list(data_clientes[['Gender', 
                                                              'Ever_Married', 
                                                              'Age', 
                                                              'Graduated', 
                                                              'Profession', 
                                                              'Work_Experience',
                                                              'Spending_Score',
                                                              'Family_Size',
                                                              'Var_1']]), 
                             'Importancia': clasificacion_ba.feature_importances_}).sort_values('Importancia', ascending=False)
  text_data = importancia.to_json(orient="table")
  json_data = json.loads(text_data)
  segmentation_data_finish['importancia'] = json_data['data']

  # Rendimiento
  y_score = clasificacion_ba.predict_proba(x_validation)
  y_test_bin = label_binarize(y_validation, classes=[0,
                                                    1, 
                                                    2, 
                                                    3])
  n_classes = y_test_bin.shape[1]
  # Se calcula la curva ROC y el área bajo la curva para cada clase
  fpr = dict()
  tpr = dict()
  response = []
  for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    response.append(auc(fpr[i], tpr[i]))
  segmentation_data_finish['auc'] = response

  return segmentation_data_finish


@router.get("/eda-improved/{id_user}/{file_name}", tags=["Improved Algorithms"])
async def get_eda_improved(id_user: int, file_name: str):
  eda_data_finish = {}

  file = db_module.crud_files.download_file(id_user=id_user, file_name=file_name)
  if file is None:
    raise HTTPException(status_code=404, detail="Archivo no encontrados ...")
  
  # Obtenemos los datos principales
  main_data = pd.read_csv('./docs/' + file_name)
  text_data = main_data.to_json(orient="table")
  json_data = json.loads(text_data)
  eda_data_finish['main_data'] = json_data['data']

  # Reemplazar datos nulos por un 99
  main_data.fillna(99, inplace=True)
  # Obtenemos los nombres de las columnas
  nombres_columnas = main_data.columns.tolist()
  for item in nombres_columnas:
    if isinstance(main_data[item].iloc[0], str):
      valores_unicos = main_data[item].unique()
      mapeo = {valor: indice + 1 for indice, valor in enumerate(valores_unicos)}
      main_data[item] = main_data[item].replace(mapeo)
  # Se crea el nuevo objeto con la ingeniería y aaplicada
  text_data = main_data.to_json(orient="table")
  json_data = json.loads(text_data)
  eda_data_finish['main_data_new'] = json_data['data']

  # Paso 1: Descripción de la estructura de datos
  eda_data_finish['shape'] = main_data.shape
  # Inicializamos objeto
  json_object = {}
  for idx, item in enumerate(main_data.dtypes):
    json_object[nombres_columnas[idx]] = item.name
  json_object["dtype"] = "object"
  eda_data_finish['d_types'] = json_object

  # Paso 2: Identificación de datos faltantes
  json_object = {}
  for idx, item in enumerate(main_data.isnull().sum()):
    json_object[nombres_columnas[idx]] = item
  json_object["dtype"] = "int64"
  eda_data_finish['is_null'] = json_object

  # Paso 3: Detección de valores atípicos
  df = pd.DataFrame.from_dict(main_data)
  text_data = df.describe().to_json(orient="table")
  json_data = json.loads(text_data)
  eda_data_finish['describe'] = json_data['data']

  # Paso 4: Identificación de relaciones entre pares variables
  df = pd.DataFrame.from_dict(main_data)
  text_data = df.corr().to_json(orient="table")
  json_data = json.loads(text_data)
  eda_data_finish['corr'] = json_data['data']

  route_local = f"./docs/{file_name}"
  os.remove(route_local)

  return eda_data_finish