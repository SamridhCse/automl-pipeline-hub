import streamlit as st
import pandas as pd
import requests
import json

API_URL = "http://127.0.0.1:8001"

st.set_page_config(page_title="AutoML Pipeline Hub", layout="wide")
st.title("⚡ AutoML Pipeline Hub")
st.markdown("Automated Preprocessing • Optuna Hyperparameter Tuning • MLflow Registry • FastAPI Inference")

tab1, tab2 = st.tabs(["🚀 1. Train New Dataset", "🎯 2. Live Model Inference"])

with tab1:
    st.subheader("Upload Dataset & Launch AutoML")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("**Dataset Preview:**", df.head(5))

        col1, col2, col3 = st.columns(3)
        with col1:
            target_col = st.selectbox("Select Target Column (y)", options=df.columns)
        with col2:
            experiment_name = st.text_input("Experiment Name", value="Customer_Churn")
        with col3:
            n_trials = st.slider("Number of Optuna Trials", min_value=3, max_value=50, value=10)

        if st.button("Start AutoML Pipeline", type="primary"):
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            data = {
                "target_col": target_col,
                "experiment_name": experiment_name,
                "n_trials": n_trials
            }
            with st.spinner("Dispatching job to FastAPI Background Worker..."):
                res = requests.post(f"{API_URL}/automl/train", files=files, data=data)
                if res.status_code == 200:
                    out = res.json()
                    st.success(f"✅ {out['message']}")
                    st.info(f"**Registered Champion Model Name:** `{out['registered_model_target']}` (Check MLflow UI at http://localhost:5000 for live trial progress!)")
                else:
                    st.error(f"Error: {res.text}")

with tab2:
    st.subheader("Query a Registered Champion Model")
    model_name = st.text_input("Registered Model Name", value="Test_Iris_champion")
    
    default_json = json.dumps([
        {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        }
    ], indent=2)
    
    raw_input = st.text_area("Input Records (JSON Array)", value=default_json, height=180)

    if st.button("Run Prediction"):
        try:
            records = json.loads(raw_input)
            res = requests.post(f"{API_URL}/automl/predict/{model_name}", json={"data": records})
            if res.status_code == 200:
                st.success("Prediction Successful!")
                st.json(res.json())
            else:
                st.error(f"API Error: {res.text}")
        except Exception as e:
            st.error(f"Invalid JSON or Connection Error: {e}")