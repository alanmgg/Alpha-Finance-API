from typing import List, Dict, Any
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

class Pronostics(BaseModel):
  column_dependient: str
  column: Dict[str, Any] = None