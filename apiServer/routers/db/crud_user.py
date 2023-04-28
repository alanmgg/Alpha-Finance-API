from firebase import firebase
from . import schemas

firebase = firebase.FirebaseApplication("https://yfinance-firebase-default-rtdb.firebaseio.com/", None)

def create_user(user: schemas.Users):
  json_user = {
    'name': user.name,
    'last_name': user.last_name,
    'email': user.email,
    'phone': user.phone,
    'password': user.password
  }
  firebase.post('/users', json_user)
  return user

def get_users():
  result = firebase.get('/users', '')
  return result

def get_user(email):
  result = get_users()
  
  for item in result:
    if result[item]['email'] == email:
      user_result = firebase.get('/users/' + item, '')

  return user_result

def delete_user(email):
  result = get_users()

  for item in result:
    if result[item]['email'] == email:
      user_result = firebase.delete('/users', item)
  
  return user_result