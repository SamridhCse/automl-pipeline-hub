import os
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

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

def instantiate_model(classifier_name: str, params: dict, is_regression: bool):
    if is_regression:
        # REGRESSION MODELS
        if classifier_name == "LinearReg":
            return LinearRegression()
        elif classifier_name == "RandomForest":
            return RandomForestRegressor(
                n_estimators=params.get("rf_n_estimators", 100),
                max_depth=params.get("rf_max_depth", 10),
                random_state=42
            )
        elif classifier_name == "SVR":
            return SVR(C=params.get("svm_c", 1.0))
        elif classifier_name == "KNN":
            return KNeighborsRegressor(n_neighbors=params.get("knn_n_neighbors", 5))
        elif classifier_name == "LightGBM":
            return LGBMRegressor(
                learning_rate=params.get("lgbm_lr", 0.1),
                num_leaves=params.get("lgbm_num_leaves", 31),
                random_state=42,
                verbose=-1
            )
        else: # XGBoost
            return XGBRegressor(
                learning_rate=params.get("xgb_lr", 0.1),
                max_depth=params.get("xgb_max_depth", 6),
                random_state=42
            )
    else:
        # CLASSIFICATION MODELS
        if classifier_name == "LogReg":
            return LogisticRegression(C=params.get("logreg_c", 1.0), max_iter=500, random_state=42)
        elif classifier_name == "RandomForest":
            return RandomForestClassifier(
                n_estimators=params.get("rf_n_estimators", 100),
                max_depth=params.get("rf_max_depth", 10),
                random_state=42
            )
        elif classifier_name == "SVM":
            return SVC(C=params.get("svm_c", 1.0), probability=True, random_state=42)
        elif classifier_name == "KNN":
            return KNeighborsClassifier(n_neighbors=params.get("knn_n_neighbors", 5))
        elif classifier_name == "LightGBM":
            return LGBMClassifier(
                learning_rate=params.get("lgbm_lr", 0.1),
                num_leaves=params.get("lgbm_num_leaves", 31),
                random_state=42,
                verbose=-1
            )
        else: # XGBoost
            return XGBClassifier(
                learning_rate=params.get("xgb_lr", 0.1),
                max_depth=params.get("xgb_max_depth", 6),
                eval_metric="logloss",
                random_state=42
            )

def run_automl_job(df: pd.DataFrame, target_col: str, experiment_name: str, n_trials: int = 10):
    X = df.drop(columns=[target_col])
    
    cols_to_drop = [col for col in X.columns if col.lower() == 'id' or col.lower().endswith('_id')]
    if cols_to_drop:
        print(f"\n[DATA CLEANING] Dropping useless ID columns: {cols_to_drop}")
        X = X.drop(columns=cols_to_drop, errors='ignore')

    y = df[target_col] 

    is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > 15
    task_type = "REGRESSION" if is_regression else "CLASSIFICATION"
    print(f"\n[DETECTED TASK] {task_type} based on target column '{target_col}'")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = build_preprocessor(X_train)

    mlflow.set_experiment(experiment_name)

    def objective(trial):
        with mlflow.start_run(nested=True):
            if is_regression:
                classifier_name = trial.suggest_categorical("model", ["LinearReg", "RandomForest", "XGBoost", "SVR", "KNN", "LightGBM"])
            else:
                classifier_name = trial.suggest_categorical("model", ["LogReg", "RandomForest", "XGBoost", "SVM", "KNN", "LightGBM"])
            
            # Hyperparameters suggest karo model ke hisaab se
            params = {}
            if classifier_name == "LogReg":
                params = {"logreg_c": trial.suggest_float("logreg_c", 1e-3, 10.0, log=True)}
            elif classifier_name in ["RandomForest", "RandomForestRegressor"]:
                params = {
                    "rf_n_estimators": trial.suggest_int("rf_n_estimators", 50, 200),
                    "rf_max_depth": trial.suggest_int("rf_max_depth", 3, 15)
                }
            elif classifier_name in ["SVM", "SVR"]:
                params = {"svm_c": trial.suggest_float("svm_c", 1e-3, 10.0, log=True)}
            elif classifier_name == "KNN":
                params = {"knn_n_neighbors": trial.suggest_int("knn_n_neighbors", 3, 15)}
            elif classifier_name == "LightGBM":
                params = {
                    "lgbm_lr": trial.suggest_float("lgbm_lr", 0.01, 0.3, log=True),
                    "lgbm_num_leaves": trial.suggest_int("lgbm_num_leaves", 10, 100)
                }
            else:
                params = {
                    "xgb_lr": trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
                    "xgb_max_depth": trial.suggest_int("xgb_max_depth", 3, 10)
                }

            model = instantiate_model(classifier_name, params, is_regression)
            full_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            scoring_metric = "r2" if is_regression else "f1_weighted"
            score = cross_val_score(full_pipeline, X_train, y_train, cv=3, scoring=scoring_metric).mean()
            
            mlflow.log_param("model_type", classifier_name)
            mlflow.log_params(params)
            mlflow.log_metric(f"cv_{scoring_metric}", score)
            return score

    with mlflow.start_run(run_name="AutoML_Master_Search"):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params.copy()
        clf_type = best_params.pop("model")
        best_model = instantiate_model(clf_type, best_params, is_regression)

        winning_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", best_model)
        ])
        winning_pipeline.fit(X_train, y_train)
        
        scoring_metric = "R2" if is_regression else "Accuracy"
        test_score = winning_pipeline.score(X_test, y_test)

        mlflow.log_param("winning_model", clf_type)
        mlflow.log_metric("test_score", test_score)

        model_name = f"{experiment_name}_champion"
        mlflow.sklearn.log_model(
            sk_model=winning_pipeline,
            name="best_pipeline", 
            registered_model_name=model_name,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )
        print(f"\n[SUCCESS] Champion Model ({clf_type}) registered as '{model_name}' with Test {scoring_metric}: {test_score:.4f}")
        return {"winner": clf_type, "test_score": test_score, "model_name": model_name}