import json

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