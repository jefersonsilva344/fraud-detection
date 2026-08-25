"""Streamlit dashboard for the FraudShield AI inference API."""

import os

import requests
import streamlit as st


API_URL = os.getenv("FRAUD_API_URL", "http://localhost:8000")

st.set_page_config(page_title="FraudShield AI", page_icon="FraudShield", layout="wide")
st.title("FraudShield AI")
st.caption("Analise de risco de transacoes com XGBoost.")

with st.sidebar:
    st.header("Conexao")
    api_url = st.text_input("URL da API", API_URL).rstrip("/")
    st.info("V1-V28 sao componentes anonimizados do dataset.")

transaction = {"Time": 0.0, "Amount": 149.62}
transaction.update({f"V{i}": 0.0 for i in range(1, 29)})

with st.form("transaction"):
    st.subheader("Transacao")
    first, second = st.columns(2)
    with first:
        transaction["Time"] = st.number_input("Time", min_value=0.0, value=0.0)
    with second:
        transaction["Amount"] = st.number_input("Amount", min_value=0.0, value=149.62)

    with st.expander("Variaveis anonimizadas (V1-V28)"):
        columns = st.columns(4)
        for index in range(1, 29):
            with columns[(index - 1) % 4]:
                transaction[f"V{index}"] = st.number_input(f"V{index}", value=0.0)

    submitted = st.form_submit_button("Analisar transacao", type="primary")

if submitted:
    try:
        response = requests.post(
            f"{api_url}/predictions/fraud",
            json={"features": transaction},
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        st.error(f"Nao foi possivel consultar a API: {exc}")
    else:
        if result["is_fraud"]:
            st.error("Fraude detectada")
        else:
            st.success("Transacao classificada como legitima")

        risk_score, threshold = st.columns(2)
        risk_score.metric("Risco de fraude", f"{result['risk_percentage']:.2f}%")
        threshold.metric("Threshold operacional", f"{result['threshold']:.2%}")
