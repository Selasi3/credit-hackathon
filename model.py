from enum import Enum, IntEnum, StrEnum
from pydantic import BaseModel

class homeEnum(StrEnum):
    RENT = 'RENT'
    MORTGAGE = "MORTGAGE"
    OWN = "OWN"
    OTHER = "OTHER"


class StatusEnum(IntEnum):
    APPROVED = 1
    DENIED = 0

class PredictionRequest(BaseModel):
    home: homeEnum
    status:StatusEnum 
    amount: float
    emp_length: float
    rate: float
    percent_income: float