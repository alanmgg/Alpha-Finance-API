from typing import Optional
from pydantic import BaseModel

class Users(BaseModel):
  name: str
  last_name: str
  email: str
  phone: int
  password: str