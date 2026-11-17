import io
import logging
import os
from functools import lru_cache

import mlflow.pyfunc
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, Path
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from app.engine import run_automl_job
from app.schemas import PredictRequest, PredictResponse, TrainResponse

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("automl.api")

app = FastAPI(
    title="AutoML Pipeline Hub",
    description="Automated preprocessing, Optuna hyperparameter tuning, and MLflow deployment API",
    version="1.0.0"
)

@lru_cache(maxsize=10)
def _load_model(model_name: str):
    logger.info(f"Loading model from MLflow: {model_name}")
    model_uri = f"models:/{model_name}/latest" 
    return mlflow.pyfunc.load_model(model_uri)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/automl/models")
def list_registered_models():
    try:
        # Database ka sahi path pass kiya
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        client = MlflowClient(tracking_uri=tracking_uri)
        # Max 1000 models fetch karega
        models = client.search_registered_models(max_results=1000)
        model_names = [m.name for m in models]
        return {"models": sorted(model_names, reverse=True)} # Latest sabse upar
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        return {"models": []}

@app.post("/automl/train", response_model=TrainResponse)
async def trigger_automl(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_col: str = Form(...),
    experiment_name: str = Form("Default_AutoML_Project"),
    n_trials: int = Form(10, ge=1, le=200)
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

    if target_col not in df.columns:
        raise HTTPException(
            status_code=400, 
            detail=f"Target column '{target_col}' not found. Available columns: {list(df.columns)}"
        )

    background_tasks.add_task(run_automl_job, df, target_col, experiment_name, n_trials)
    
    return TrainResponse(
        message="AutoML pipeline started in the background.",
        experiment_name=experiment_name,
        registered_model_target=f"{experiment_name}_champion",
        n_trials=n_trials
    )

@app.post("/automl/predict/{model_name}", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    model_name: str = Path(..., pattern=r"^[A-Za-z0-9_.\-]{1,64}$")
):
    try:
        loaded_pipeline = _load_model(model_name)
    except MlflowException as e:
        logger.error(f"MLflow loading failed: {str(e)}")
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_name}' failed to load. Error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"MLflow connection failed: {e}")
        raise HTTPException(status_code=503, detail="MLflow service unavailable.")

    try:
        input_df = pd.DataFrame(request.data)

        sig = loaded_pipeline.metadata.signature
        if sig and sig.inputs and not sig.inputs.is_tensor_spec():
            expected_cols = sig.inputs.input_names()
            try:
                input_df = input_df[expected_cols]
            except KeyError:
                raise HTTPException(
                    status_code=422, 
                    detail=f"Missing required features. Expected: {expected_cols}"
                )

        predictions = loaded_pipeline.predict(input_df)
        
        return PredictResponse(
            model_name=model_name,
            predictions=predictions.tolist()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error.")

@app.get("/", include_in_schema=False)
def root():
    return {"service": "automl-pipeline-hub", "docs": "/docs", "health": "/health"}