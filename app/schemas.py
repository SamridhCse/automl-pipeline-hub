from pydantic import BaseModel, Field
from typing import List, Dict, Any

class TrainResponse(BaseModel):
    message: str
    experiment_name: str
    registered_model_target: str
    n_trials: int

class PredictRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(
        ..., 
        description="List of records (dictionaries) representing rows for inference."
    )

class PredictResponse(BaseModel):
    model_name: str
    predictions: List[Any]