from firebase import firebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

firebase = firebase.FirebaseApplication("https://yfinance-firebase-default-rtdb.firebaseio.com/", None)
cred = credentials.Certificate("./apiServer/routers/db/key/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
  'storageBucket': 'firebas-api.appspot.com'
})

bucket = storage.bucket()

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
        names_files.append(blob.name)
      return names_files
  
  return None

def download_file(id_user, file_name):
  result = firebase.get('/users', '')
  for item in result:
    if result[item]['id_user'] == id_user:
      user_result = firebase.get('/users/' + item, '')
      route_remote = f"{user_result['email']}/{file_name}"
      blob = bucket.blob(route_remote)
      contenido_archivo = blob.download_as_bytes()
      return contenido_archivo
  
  return None