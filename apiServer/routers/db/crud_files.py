from firebase import firebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from datetime import datetime
import os

firebase = firebase.FirebaseApplication("https://yfinance-firebase-default-rtdb.firebaseio.com/", None)
cred = credentials.Certificate("./apiServer/routers/db/key/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
  'storageBucket': 'firebas-api.appspot.com'
})

bucket = storage.bucket()

meses = {
  1: "Ene",
  2: "Feb",
  3: "Mar",
  4: "Abr",
  5: "May",
  6: "Jun",
  7: "Jul",
  8: "Ago",
  9: "Sep",
  10: "Oct",
  11: "Nov",
  12: "Dic"
}

def upload_file(id_user, filename, route_local):
  result = firebase.get('/users', '')
  for item in result:
    if result[item]['id_user'] == id_user:
      user_result = firebase.get('/users/' + item, '')
      route_remote = f"{user_result['email']}/{filename}"
      blob = bucket.blob(route_remote)
      blob.upload_from_filename(route_local)

      return user_result
  
  return None

def get_files(id_user):
  result = firebase.get('/users', '')
  for item in result:
    if result[item]['id_user'] == id_user:
      user_result = firebase.get('/users/' + item, '')
      route_remote = f"{user_result['email']}"
      blobs = bucket.list_blobs(prefix=route_remote)
      names_files = []
      for blob in blobs:
        name = blob.name.split('/')
        size = round(blob.size / (1024), 2)
        month = meses[blob.updated.month]
        date = blob.updated.strftime("%d {} %Y".format(month))

        names_files.append({ 'name': name[1], 'size': str(size) + ' KB', 'type': blob.content_type, 'date': date })

      return names_files
  
  return None

def download_file(id_user, file_name):
  result = firebase.get('/users', '')
  for item in result:
    if result[item]['id_user'] == id_user:
      user_result = firebase.get('/users/' + item, '')
      route_remote = f"{user_result['email']}/{file_name}"
      blob = bucket.blob(route_remote)
      archivo_local = os.path.join('./docs', file_name)
      blob.download_to_filename(archivo_local)

      return { "message": "Archivo descargado y creado localmente" }
  
  return None

def delete_file(id_user, file_name):
  result = firebase.get('/users', '')
  for item in result:
    if result[item]['id_user'] == id_user:
      user_result = firebase.get('/users/' + item, '')
      route_remote = f"{user_result['email']}/{file_name}"
      blob = bucket.blob(route_remote)
      try:
        blob.delete()
        return { "message": "Archivo eliminado" }
      except:
        return None
  
  return None