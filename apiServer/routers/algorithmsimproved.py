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


@router.get("/main-data/{id_user}/{file_name}", tags=["Improved Algorithms"])
async def get_main_data(id_user: int, file_name: str):
  data_finish = {}

  file = db_module.crud_files.download_file(id_user=id_user, file_name=file_name)
  if file is None:
    raise HTTPException(status_code=404, detail="Archivo no encontrados ...")
  
  # Obtenemos los datos principales
  main_data = pd.read_csv('./docs/' + file_name)
  text_data = main_data.to_json(orient="table")
  json_data = json.loads(text_data)
  data_finish['main_data'] = json_data['data']

  route_local = f"./docs/{file_name}"
  os.remove(route_local)

  return data_finish


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
  eda_data_finish['corr_data'] = json_data['data']

  route_local = f"./docs/{file_name}"
  os.remove(route_local)

  return eda_data_finish


@router.get("/acp-improved/{id_user}/{file_name}", tags=["Improved Algorithms"])
async def get_acp_improved(id_user: int, file_name: str):
  acp_data_finish = {}

  file = db_module.crud_files.download_file(id_user=id_user, file_name=file_name)
  if file is None:
    raise HTTPException(status_code=404, detail="Archivo no encontrados ...")
  
  # Obtenemos los datos principales
  main_data = pd.read_csv('./docs/' + file_name)
  text_data = main_data.to_json(orient="table")
  json_data = json.loads(text_data)
  acp_data_finish['main_data'] = json_data['data']

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
  acp_data_finish['main_data_new'] = json_data['data']

  # Paso 1: Hay evidencia de variables posiblemente correlacionadas
  corr_data = main_data.corr(method='pearson')
  df = pd.DataFrame.from_dict(corr_data)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  acp_data_finish['corr_data'] = json_data['data']

  # Paso 2: Se hace una estandarización de los datos
  estandarizar = StandardScaler()
  m_estandarizada = estandarizar.fit_transform(main_data)
  df = pd.DataFrame(m_estandarizada, columns=main_data.columns)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  acp_data_finish['m_estandarizada'] = json_data['data']

  # Pasos 3 y 4: Se calcula la matriz de covarianzas o correlaciones, y se 
  # calculan los componentes (eigen-vectores) y la varianza (eigen-valores)
  pca = PCA(n_components=len(nombres_columnas))
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

  cargas_components = pd.DataFrame(abs(pca.components_), columns=main_data.columns)
  text_data = cargas_components.to_json(orient="table")
  json_data = json.loads(text_data)
  acp_data_finish['cargas_components'] = json_data['data']

  route_local = f"./docs/{file_name}"
  os.remove(route_local)

  return acp_data_finish


@router.get("/forecast-ad-improved/{id_user}/{file_name}/{column_dependient}", tags=["Improved Algorithms"])
async def get_forecast_ad_improved(id_user: int, file_name: str, column_dependient: str):
  ad_data_finish = {}

  file = db_module.crud_files.download_file(id_user=id_user, file_name=file_name)
  if file is None:
    raise HTTPException(status_code=404, detail="Archivo no encontrados ...")
  
  # Obtenemos los datos principales
  main_data = pd.read_csv('./docs/' + file_name)
  text_data = main_data.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_data_finish['main_data'] = json_data['data']

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
  ad_data_finish['main_data_new'] = json_data['data']

  json_object = {}
  for idx, item in enumerate(main_data.isnull().sum()):
    json_object[nombres_columnas[idx]] = item
  json_object["dtype"] = "int64"
  ad_data_finish['is_null'] = json_object

  df = pd.DataFrame.from_dict(main_data)
  text_data = df.describe().to_json(orient="table")
  json_data = json.loads(text_data)
  ad_data_finish['describe'] = json_data['data']

  # Aplicación del algoritmo
  nombres_columnas.remove(column_dependient)
  x = np.array(main_data[nombres_columnas])
  df = pd.DataFrame.from_dict(x)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_data_finish['x'] = json_data['data']

  y = np.array(main_data[[column_dependient]])
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
  
  importancia = pd.DataFrame({'Variable': list(main_data[nombres_columnas]),
                              'Importancia': pronostico_ad.feature_importances_})
  text_data = importancia.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_data_finish['importancia'] = json_data['data']

  ad_data_finish['columnas'] = nombres_columnas

  route_local = f"./docs/{file_name}"
  os.remove(route_local)

  return ad_data_finish


@router.get("/forecast-ad-pronostic-improved/{id_user}/{file_name}", tags=["Improved Algorithms"])
async def get_forecast_ad_pronostic_improved(pronostic: db_module.schemas.Pronostics, id_user: int, file_name: str):
  ad_data_pronostic_finish = {}
  new_pronostic = {}

  for item in pronostic.column:
    new_pronostic[item] = [pronostic.column[item]]

  file = db_module.crud_files.download_file(id_user=id_user, file_name=file_name)
  if file is None:
    raise HTTPException(status_code=404, detail="Archivo no encontrados ...")
  
  # Obtenemos los datos principales
  main_data = pd.read_csv('./docs/' + file_name)

  # Reemplazar datos nulos por un 99
  main_data.fillna(99, inplace=True)
  # Obtenemos los nombres de las columnas
  nombres_columnas = main_data.columns.tolist()
  for item in nombres_columnas:
    if isinstance(main_data[item].iloc[0], str):
      valores_unicos = main_data[item].unique()
      mapeo = {valor: indice + 1 for indice, valor in enumerate(valores_unicos)}
      main_data[item] = main_data[item].replace(mapeo)

  # Aplicación del algoritmo
  nombres_columnas.remove(pronostic.column_dependient)
  x = np.array(main_data[nombres_columnas])

  y = np.array(main_data[[pronostic.column_dependient]])

  # Se hace la división de los datos
  x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 0, 
                                                                    shuffle = True)

  # Se entrena el modelo
  pronostico_ad = DecisionTreeRegressor(random_state=0)
  pronostico_ad.fit(x_train, y_train)

  # Nuevo pronostico
  pronostic_model = pd.DataFrame(new_pronostic)
  ad_data_pronostic_finish['pronostic'] = pronostico_ad.predict(pronostic_model)[0]

  route_local = f"./docs/{file_name}"
  os.remove(route_local)

  return ad_data_pronostic_finish


@router.get("/forecast-ba-improved/{id_user}/{file_name}/{column_dependient}", tags=["Improved Algorithms"])
async def get_forecast_ba_improved(id_user: int, file_name: str, column_dependient: str):
  ba_data_finish = {}

  file = db_module.crud_files.download_file(id_user=id_user, file_name=file_name)
  if file is None:
    raise HTTPException(status_code=404, detail="Archivo no encontrados ...")
  
  # Obtenemos los datos principales
  main_data = pd.read_csv('./docs/' + file_name)
  text_data = main_data.to_json(orient="table")
  json_data = json.loads(text_data)
  ba_data_finish['main_data'] = json_data['data']

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
  ba_data_finish['main_data_new'] = json_data['data']

  # Descripción de la estructura de los datos
  json_object = {}
  for idx, item in enumerate(main_data.isnull().sum()):
    json_object[nombres_columnas[idx]] = item
  json_object["dtype"] = "int64"
  ba_data_finish['is_null'] = json_object

  df = pd.DataFrame.from_dict(main_data)
  text_data = df.describe().to_json(orient="table")
  json_data = json.loads(text_data)
  ba_data_finish['describe'] = json_data['data']

  # Aplicación del algoritmo
  nombres_columnas.remove(column_dependient)
  x = np.array(main_data[nombres_columnas])
  df = pd.DataFrame.from_dict(x)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  ba_data_finish['x'] = json_data['data']

  y = np.array(main_data[[column_dependient]])
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
  
  importancia = pd.DataFrame({'Variable': list(main_data[nombres_columnas]),
                              'Importancia': pronostico_ba.feature_importances_})
  text_data = importancia.to_json(orient="table")
  json_data = json.loads(text_data)
  ba_data_finish['importancia'] = json_data['data']

  route_local = f"./docs/{file_name}"
  os.remove(route_local)

  return ba_data_finish


@router.get("/forecast-ba-pronostic-improved/{id_user}/{file_name}", tags=["Improved Algorithms"])
async def get_forecast_ba_pronostic_improved(pronostic: db_module.schemas.Pronostics, id_user: int, file_name: str):
  ba_data_pronostic_finish = {}
  new_pronostic = {}

  for item in pronostic.column:
    new_pronostic[item] = [pronostic.column[item]]

  file = db_module.crud_files.download_file(id_user=id_user, file_name=file_name)
  if file is None:
    raise HTTPException(status_code=404, detail="Archivo no encontrados ...")
  
  # Obtenemos los datos principales
  main_data = pd.read_csv('./docs/' + file_name)

  # Reemplazar datos nulos por un 99
  main_data.fillna(99, inplace=True)
  # Obtenemos los nombres de las columnas
  nombres_columnas = main_data.columns.tolist()
  for item in nombres_columnas:
    if isinstance(main_data[item].iloc[0], str):
      valores_unicos = main_data[item].unique()
      mapeo = {valor: indice + 1 for indice, valor in enumerate(valores_unicos)}
      main_data[item] = main_data[item].replace(mapeo)

  # Aplicación del algoritmo
  nombres_columnas.remove(pronostic.column_dependient)
  x = np.array(main_data[nombres_columnas])

  y = np.array(main_data[[pronostic.column_dependient]])

  # Se hace la división de los datos
  x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 0, 
                                                                    shuffle = True)

  # Se entrena el modelo
  pronostico_ba = RandomForestRegressor(random_state=0)
  pronostico_ba.fit(x_train, y_train)

  # Nuevo pronostico
  pronostic_model = pd.DataFrame(new_pronostic)
  ba_data_pronostic_finish['pronostic'] = pronostico_ba.predict(pronostic_model)[0]

  route_local = f"./docs/{file_name}"
  os.remove(route_local)

  return ba_data_pronostic_finish


@router.get("/classification-ad-ba-improved/{id_user}/{file_name}/{column_dependient}", tags=["Improved Algorithms"])
async def get_forecast_ad_ba(id_user: int, file_name: str, column_dependient: str):
  ad_ba_data_finish = {}

  file = db_module.crud_files.download_file(id_user=id_user, file_name=file_name)
  if file is None:
    raise HTTPException(status_code=404, detail="Archivo no encontrados ...")
  
  # Obtenemos los datos principales
  main_data = pd.read_csv('./docs/' + file_name)
  text_data = main_data.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_ba_data_finish['main_data'] = json_data['data']

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
  ad_ba_data_finish['main_data_new'] = json_data['data']

  # Descripción de la estructura de los datos
  json_object = {}
  for idx, item in enumerate(main_data.isnull().sum()):
    json_object[nombres_columnas[idx]] = item
  json_object["dtype"] = "int64"
  ad_ba_data_finish['is_null'] = json_object

  df = pd.DataFrame.from_dict(main_data)
  text_data = df.describe().to_json(orient="table")
  json_data = json.loads(text_data)
  ad_ba_data_finish['describe'] = json_data['data']

  text_data = main_data.groupby(column_dependient).size().to_json(orient="table")
  json_data = json.loads(text_data)
  ad_ba_data_finish['groupby'] = json_data['data']

  text_data = main_data.corr().to_json(orient="table")
  json_data = json.loads(text_data)
  ad_ba_data_finish['corr_data'] = json_data['data']

  # Aplicación del algoritmo
  nombres_columnas.remove(column_dependient)
  x = np.array(main_data[nombres_columnas])
  df = pd.DataFrame(x)
  text_data = df.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_ba_data_finish['x'] = json_data['data']

  # Variable clase
  y = np.array(main_data[[column_dependient]])
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
  
  importancia_mod_1 = pd.DataFrame({'Variable': list(main_data[nombres_columnas]),
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

  importancia_mod_2 = pd.DataFrame({'Variable': list(main_data[nombres_columnas]), 
                             'Importancia': clasificacion_ba.feature_importances_}).sort_values('Importancia', ascending=False)
  text_data = importancia_mod_2.to_json(orient="table")
  json_data = json.loads(text_data)
  ad_ba_data_finish['importancia_mod_2'] = json_data['data']

  ad_ba_data_finish['validation'] = { 'ad': accuracy_score(y_validation, y_clasificacion_ad), 
                                      'ba': accuracy_score(y_validation, y_clasificacion_ba) }

  return ad_ba_data_finish


@router.get("/classification-ad-ba-pronostic-improved/{id_user}/{file_name}", tags=["Improved Algorithms"])
async def get_forecast_ad_ba(pronostic: db_module.schemas.Pronostics, id_user: int, file_name: str):
  ad_ba_data_pronostic_finish = {}
  new_pronostic = {}

  for item in pronostic.column:
    new_pronostic[item] = [pronostic.column[item]]

  file = db_module.crud_files.download_file(id_user=id_user, file_name=file_name)
  if file is None:
    raise HTTPException(status_code=404, detail="Archivo no encontrados ...")
  
  # Obtenemos los datos principales
  main_data = pd.read_csv('./docs/' + file_name)

  # Reemplazar datos nulos por un 99
  main_data.fillna(99, inplace=True)
  # Obtenemos los nombres de las columnas
  nombres_columnas = main_data.columns.tolist()
  for item in nombres_columnas:
    if isinstance(main_data[item].iloc[0], str):
      valores_unicos = main_data[item].unique()
      mapeo = {valor: indice + 1 for indice, valor in enumerate(valores_unicos)}
      main_data[item] = main_data[item].replace(mapeo)

  # Aplicación del algoritmo
  nombres_columnas.remove(pronostic.column_dependient)
  x = np.array(main_data[nombres_columnas])

  # Variable clase
  y = np.array(main_data[[pronostic.column_dependient]])

  x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)

  # Se entrena el modelo a partir de los datos de entrada
  clasificacion_ad = DecisionTreeClassifier(random_state=0)
  clasificacion_ad = clasificacion_ad.fit(x_train, y_train)

  clasificacion_ba = RandomForestClassifier(random_state=0)
  clasificacion_ba = clasificacion_ba.fit(x_train, y_train)
  
  pronostic_model = pd.DataFrame(new_pronostic)
  classifcation = int(clasificacion_ba.predict(pronostic_model)[0])
  ad_ba_data_pronostic_finish['pronostic'] = classifcation

  route_local = f"./docs/{file_name}"
  os.remove(route_local)

  return ad_ba_data_pronostic_finish