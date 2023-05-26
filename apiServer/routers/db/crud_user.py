from firebase import firebase
from . import schemas

firebase = firebase.FirebaseApplication("https://yfinance-firebase-default-rtdb.firebaseio.com/", None)

def create_user(user: schemas.Users):
  result = firebase.get('/users', '')
  if result == None:
    id_user = 1
  else:
    id_user = len(result) + 1

  json_user = {
    'id_user': id_user,
    'name': user.name,
    'last_name': user.last_name,
    'email': user.email,
    'phone': user.phone,
    'password': user.password,
  }
  firebase.post('/users', json_user)
  return user

def get_users():
  result = firebase.get('/users', '')
  return result

def get_user(id_user):
  result = firebase.get('/users', '')
  for item in result:
    if result[item]['id_user'] == id_user:
      user_result = firebase.get('/users/' + item, '')
      return user_result
  
  return None

def get_user_by_email(email):
  result = firebase.get('/users', '')
  for item in result:
    if result[item]['email'] == email:
      user_result = firebase.get('/users/' + item, '')
      return user_result

  return None

def delete_user(id_user):
  result = firebase.get('/users', '')
  for item in result:
    if result[item]['id_user'] == id_user:
      firebase.delete('/users', item)
      return result[item]['name']
  
  return None

def update_user(id_user, user_companies):
  result = firebase.get('/users', '')
  for item in result:
    if result[item]['id_user'] == id_user:
      firebase.put('/users/' + item, 'companies', user_companies.companies)
      return result[item]['name']
  
  return None