import streamlit as st
import pandas as pd
import joblib
import numpy as np

try:
    model = joblib.load('decision_tree_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
except:
    st.error("Model dosyası bulunamadı! Lütfen önce proje.ipynb dosyasındaki kaydetme kodunu çalıştırın.")
    st.stop()

st.set_page_config(page_title="Kredi Risk Analizi", page_icon="🏦")

st.title("🏦 Kredi Risk Tahmin Sistemi")
st.markdown("**Maltepe Üniversitesi - Yazılım Müh. - CEN 416 Final Projesi**")
st.write("Müşteri bilgilerini girerek kredi risk durumunu (Verilir/Verilmez) tahmin edebilirsiniz.")

st.sidebar.header("Müşteri Bilgileri")

age = st.sidebar.number_input("Yaş", min_value=18, max_value=100, value=25)
income = st.sidebar.number_input("Yıllık Gelir (TL)", min_value=0, value=50000)
emp_length = st.sidebar.slider("Çalışma Süresi (Yıl)", 0, 40, 5)
loan_amount = st.sidebar.number_input("İstenen Kredi Miktarı", min_value=0, value=10000)
int_rate = st.sidebar.number_input("Faiz Oranı (%)", min_value=0.0, value=10.0)

home_ownership = st.sidebar.selectbox("Ev Durumu", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
loan_intent = st.sidebar.selectbox("Kredi Amacı", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
loan_grade = st.sidebar.selectbox("Kredi Derecesi", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
default_history = st.sidebar.selectbox("Daha önce temerrüde düştü mü?", ['Y', 'N'])

if st.button("Risk Analizi Yap"):
    
    input_data = pd.DataFrame({
        'person_age': [age],
        'person_income': [income],
        'person_emp_length': [emp_length],
        'loan_amnt': [loan_amount],
        'loan_int_rate': [int_rate],
        'loan_percent_income': [loan_amount / income if income > 0 else 0],
        'cb_person_cred_hist_length': [2],
        'person_home_ownership': [home_ownership],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'cb_person_default_on_file': [default_history]
    })

    input_data = pd.get_dummies(input_data)

    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1] 

    st.divider()
    if prediction[0] == 0:
        st.success("✅ **KREDİ ONAYLANDI** (Düşük Risk)")
        st.balloons()
    else:
        st.error("❌ **KREDİ REDDEDİLDİ** (Yüksek Risk)")
    
    st.info(f"Yapay Zekanın Risk Hesaplaması: %{probability*100:.2f}")