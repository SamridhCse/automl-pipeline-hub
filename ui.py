import os
import streamlit as st
import requests
import pandas as pd
import json

API_URL = os.getenv("API_URL", "http://127.0.0.1:8010")

st.set_page_config(page_title="AutoML Pipeline Hub", page_icon="🤖", layout="wide")
st.title("🤖 AutoML Pipeline Hub")
st.markdown("Automated Preprocessing, Optuna Tuning & MLflow Tracking")

if "sample_json" not in st.session_state:
    st.session_state.sample_json = json.dumps([{"tenure_months": 12, "total_orders": 10, "total_spend": 269.41, "support_calls": 2, "last_login_days_ago": 27}], indent=2)

tab1, tab2 = st.tabs(["🚀 Train Model", "🔮 Make Predictions"])

# ==========================================
# TAB 1: TRAIN MODEL
# ==========================================
with tab1:
    st.header("1. Upload Data & Train")
    
    uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("**Data Preview:**")
        st.dataframe(df.head())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            target_col = st.selectbox("Select Target Column", df.columns)
        with col2:
            exp_name = st.text_input("Experiment Name", value="Ecommerce_Churn")
        with col3:
            n_trials = st.number_input("Number of Optuna Trials", min_value=1, max_value=100, value=10)
        
        if st.button("Start Training 🚀", type="primary"):
            try:
                sample_row = df.drop(columns=[target_col]).iloc[0].to_dict()
                sample_row = {k: v for k, v in sample_row.items() if k.lower() != 'id' and not k.lower().endswith('_id')}
                st.session_state.sample_json = json.dumps([sample_row], indent=2, default=str)
            except Exception:
                pass

            with st.spinner("Sending data to backend..."):
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
                data = {
                    "target_col": target_col,
                    "experiment_name": exp_name,
                    "n_trials": n_trials
                }
                
                try:
                    response = requests.post(f"{API_URL}/automl/train", files=files, data=data)
                    if response.status_code == 200:
                        st.success("Training started! Check terminal for logs.")
                        st.info("💡 Wait for the [SUCCESS] message in terminal, then click 'Refresh Models' in the Predict tab.")
                    else:
                        st.error(f"Error: {response.json()}")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

# ==========================================
# TAB 2: PREDICT
# ==========================================
with tab2:
    st.header("2. Predict with Champion Model")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Refresh button
        if st.button("🔄 Refresh Model List"):
            pass # Click karne pe Streamlit apne aap script rerun karega

        try:
            res = requests.get(f"{API_URL}/automl/models")
            if res.status_code == 200:
                available_models = res.json().get("models", [])
            else:
                available_models = []
        except Exception:
            available_models = []

        if available_models:
            model_name = st.selectbox("Select Trained Model", available_models)
        else:
            st.warning("No models found. Train a model first.")
            model_name = st.text_input("Model Name", value="Ecommerce_Churn_champion")

        st.markdown("**Input Features (JSON Format):**")
        json_input = st.text_area("Enter JSON array of records", value=st.session_state.sample_json, height=250)
        
        predict_btn = st.button("Predict 🔮", type="primary")

    with col2:
        if predict_btn:
            if not model_name:
                st.error("Please select a model.")
            else:
                try:
                    parsed_data = json.loads(json_input)
                    payload = {"data": parsed_data}
                    
                    with st.spinner("Getting predictions..."):
                        response = requests.post(f"{API_URL}/automl/predict/{model_name}", json=payload)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success("Prediction Successful!")
                            
                            res_df = pd.DataFrame(parsed_data)
                            res_df["Prediction"] = result["predictions"]
                            st.dataframe(res_df)
                        else:
                            error_detail = response.json().get('detail', 'Unknown error')
                            st.error(f"Error {response.status_code}: {error_detail}")
                except json.JSONDecodeError:
                    st.warning("Invalid JSON format.")
                except Exception as e:
                    st.error(f"Connection failed: {e}")