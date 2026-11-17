import os
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_URI)

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["int64", "float64", "number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers)

def instantiate_model(classifier_name: str, params: dict):
    if classifier_name == "LogReg":
        return LogisticRegression(C=params["logreg_c"], max_iter=500, random_state=42)
    elif classifier_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=params["rf_n_estimators"],
            max_depth=params["rf_max_depth"],
            random_state=42
        )
    else:
        return XGBClassifier(
            learning_rate=params["xgb_lr"],
            max_depth=params["xgb_max_depth"],
            eval_metric="logloss",
            random_state=42
        )

def run_automl_job(df: pd.DataFrame, target_col: str, experiment_name: str, n_trials: int = 10):
    X = df.drop(columns=[target_col])
    y_raw = df[target_col]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = build_preprocessor(X_train)

    mlflow.set_experiment(experiment_name)

    def objective(trial):
        with mlflow.start_run(nested=True):
            classifier_name = trial.suggest_categorical("classifier", ["LogReg", "RandomForest", "XGBoost"])
            
            if classifier_name == "LogReg":
                params = {"logreg_c": trial.suggest_float("logreg_c", 1e-3, 10.0, log=True)}
            elif classifier_name == "RandomForest":
                params = {
                    "rf_n_estimators": trial.suggest_int("rf_n_estimators", 50, 200),
                    "rf_max_depth": trial.suggest_int("rf_max_depth", 3, 15)
                }
            else:
                params = {
                    "xgb_lr": trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
                    "xgb_max_depth": trial.suggest_int("xgb_max_depth", 3, 10)
                }

            model = instantiate_model(classifier_name, params)
            full_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", model)
            ])

            score = cross_val_score(full_pipeline, X_train, y_train, cv=3, scoring="f1_weighted").mean()
            
            mlflow.log_param("classifier", classifier_name)
            mlflow.log_params(params)
            mlflow.log_metric("cv_f1_weighted", score)
            return score

    with mlflow.start_run(run_name="AutoML_Master_Search"):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params.copy()
        clf_type = best_params.pop("classifier")
        best_model = instantiate_model(clf_type, best_params)

        winning_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", best_model)
        ])
        winning_pipeline.fit(X_train, y_train)
        test_accuracy = winning_pipeline.score(X_test, y_test)

        mlflow.log_param("winning_classifier", clf_type)
        mlflow.log_metric("best_cv_f1", study.best_value)
        mlflow.log_metric("test_accuracy", test_accuracy)

        model_name = f"{experiment_name}_champion"
        mlflow.sklearn.log_model(
            sk_model=winning_pipeline,
            name="best_pipeline",
            registered_model_name=model_name,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )
        print(f"\n[SUCCESS] Champion Model ({clf_type}) registered as '{model_name}' with Test Accuracy: {test_accuracy:.4f}")
        return {"winner": clf_type, "test_accuracy": test_accuracy, "model_name": model_name}
