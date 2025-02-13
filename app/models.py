from typing import Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.1


class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
