import io
import mlflow.pyfunc
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from app.engine import run_automl_job
from app.schemas import PredictRequest, PredictResponse, TrainResponse

app = FastAPI(
    title="AutoML Pipeline Hub",
    description="Automated preprocessing, Optuna hyperparameter tuning, and MLflow deployment API",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/automl/train", response_model=TrainResponse)
async def trigger_automl(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_col: str = Form(...),
    experiment_name: str = Form("Default_AutoML_Project"),
    n_trials: int = Form(10)
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

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
async def predict(model_name: str, request: PredictRequest):
    try:
        model_uri = f"models:/{model_name}/latest"
        loaded_pipeline = mlflow.pyfunc.load_model(model_uri)
        input_df = pd.DataFrame(request.data)
        predictions = loaded_pipeline.predict(input_df)
        return PredictResponse(
            model_name=model_name,
            predictions=predictions.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Inference failed: {str(e)}")
