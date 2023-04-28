from typing import List
from pydantic import BaseModel

class Status(BaseModel):
  message: str

class Users(BaseModel):
  name: str
  last_name: str
  email: str
  phone: int
  password: str

class UsersCompanies(BaseModel):
  companies: List[str] = []