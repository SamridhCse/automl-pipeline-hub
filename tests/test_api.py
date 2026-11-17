import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_train_missing_target_column():
    csv_content = b"feature1,feature2\n1.0,2.0\n3.0,4.0"
    response = client.post(
        "/automl/train",
        files={"file": ("dummy.csv", csv_content, "text/csv")},
        data={"target_col": "wrong_col", "experiment_name": "Test_Exp", "n_trials": 2}
    )
    assert response.status_code == 400
    assert "not found" in response.json()["detail"]

# --- FIXED: Mocked MLflow model for CI/CD ---
@patch("app.main._load_model")
def test_live_prediction_on_iris_champion(mock_load_model):
    # 1. Ek fake model banao jo fake prediction return kare
    mock_pipeline = MagicMock()
    mock_pipeline.metadata.signature = None # Signature check skip karne ke liye
    mock_pipeline.predict.return_value = np.array(["Setosa"]) # Fake prediction
    mock_load_model.return_value = mock_pipeline

    payload = {
        "data": [
            {
                "sepal length (cm)": 5.1,
                "sepal width (cm)": 3.5,
                "petal length (cm)": 1.4,
                "petal width (cm)": 0.2
            }
        ]
    }
    response = client.post("/automl/predict/Test_Iris_champion", json=payload)
    
    assert response.status_code == 200
    result = response.json()
    assert result["model_name"] == "Test_Iris_champion"
    assert len(result["predictions"]) == 1
    assert result["predictions"][0] == "Setosa" # Check karo ki fake prediction hi aaya