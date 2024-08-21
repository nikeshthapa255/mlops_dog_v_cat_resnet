from typing import Any, List, Optional

from pydantic import BaseModel


class PredictionResults(BaseModel):
    errors: Optional[str]
    version: str
    #predictions: Optional[List[int]]
    predictions: str
