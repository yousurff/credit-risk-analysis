import streamlit as st
import pandas as pd
import joblib
import numpy as np
from fpdf import FPDF
import xgboost as xgb

st.set_page_config(page_title="Kredi Risk Analizi", layout="wide")

def create_pdf(input_data, prediction, probability, risk_text, model_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="KREDI RISK ANALIZ RAPORU", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Model: {model_name}", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Musteri Bilgileri:", ln=True, align='L')
    pdf.set_font("Arial", size=11)
    
    for key, value in input_data.items():
        clean_key = str(key).replace('ı', 'i').replace('ş', 's').replace('ğ', 'g').replace('Ü', 'U').replace('Ö', 'O').replace('Ç', 'C').replace('İ', 'I')
        clean_val = str(value).replace('ı', 'i').replace('ş', 's').replace('ğ', 'g').replace('ü', 'u').replace('ö', 'o').replace('ç', 'c').replace('İ', 'I')
        pdf.cell(200, 8, txt=f"{clean_key}: {clean_val}", ln=True, align='L')
    
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="ANALIZ SONUCU:", ln=True, align='L')
    
    if prediction == 0:
        pdf.set_text_color(0, 128, 0)
        pdf.cell(200, 10, txt="DURUM: ONAYLANDI (DUSUK RISK)", ln=True)
    else:
        pdf.set_text_color(255, 0, 0)
        pdf.cell(200, 10, txt="DURUM: REDDEDILDI (YUKSEK RISK)", ln=True)
        
    pdf.set_text_color(0, 0, 0)
    pdf.cell(200, 10, txt=f"Risk Skoru: %{probability*100:.2f}", ln=True)
    
    return pdf.output(dest='S').encode('latin-1')

@st.cache_resource
def load_resources():
    models = {}
    cols = None
    try:
        models["Decision Tree"] = joblib.load('decision_tree_model.pkl')
    except: pass
    
    try:
        models["Random Forest"] = joblib.load('random_forest_model.pkl')
    except: pass

    try:
        models["XGBoost"] = joblib.load('xgboost_model.pkl')
    except: pass

    try:
        models["Logistic Regression"] = joblib.load('logistic_model.pkl')
    except: pass

    try:
        cols = joblib.load('model_columns.pkl')
    except: pass
    
    return models, cols

loaded_models, model_columns = load_resources()

stats_available = False
avg_income = 0
avg_loan_amnt = 0
avg_int_rate = 0

try:
    df_orj = pd.read_csv('credit_risk_dataset.csv')
    df_orj = df_orj[df_orj['person_age'] < 100]
    df_approved = df_orj[df_orj['loan_status'] == 0]
    avg_income = df_approved['person_income'].mean()
    avg_loan_amnt = df_approved['loan_amnt'].mean()
    avg_int_rate = df_approved['loan_int_rate'].mean()
    stats_available = True
except:
    pass

st.title("Kredi Risk Tahmin Sistemi")
st.markdown("**Maltepe Üniversitesi - Yazılım Müh. - CEN 416 Final Projesi**")

if not loaded_models or model_columns is None:
    st.error("Model dosyalari bulunamadi. Lutfen once proje.ipynb dosyasini calistirin.")
    st.stop()

col_main1, col_main2 = st.columns([1, 2])

st.sidebar.header("Model Ayarlari")
selected_model_name = st.sidebar.selectbox("Yapay Zeka Modeli", list(loaded_models.keys()))
current_model = loaded_models[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.header("Musteri Bilgileri")

age = st.sidebar.number_input("Yas", 18, 100, 25)
income = st.sidebar.number_input("Yillik Gelir (TL)", 0, value=50000, step=1000)
emp_length = st.sidebar.slider("Calisma Suresi (Yil)", 0.0, 40.0, 5.0)
loan_amount = st.sidebar.number_input("Istenen Kredi Miktari", 0, value=10000, step=500)
int_rate = st.sidebar.number_input("Faiz Orani (%)", 0.0, value=10.0, step=0.1)

home_ownership = st.sidebar.selectbox("Ev Durumu", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
loan_intent = st.sidebar.selectbox("Kredi Amaci", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
loan_grade = st.sidebar.selectbox("Kredi Derecesi", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
default_history = st.sidebar.selectbox("Daha once temerrude dustu mu?", ['Y', 'N'])

if st.button("Risk Analizi Yap", type="primary"):
    
    raw_input = {
        'Yas': age, 'Gelir': income, 'Calisma Suresi': emp_length, 
        'Kredi Miktari': loan_amount, 'Faiz': int_rate, 
        'Ev': home_ownership, 'Amac': loan_intent, 'Gecmis Temerrut': default_history
    }
    
    input_df = pd.DataFrame({
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

    input_dummies = pd.get_dummies(input_df)
    input_final = input_dummies.reindex(columns=model_columns, fill_value=0)

    prediction = current_model.predict(input_final)[0]
    probability = current_model.predict_proba(input_final)[0][1]

    st.divider()
    c1, c2 = st.columns([2, 1])
    
    risk_text = ""
    with c1:
        if prediction == 0:
            st.success("KREDİ ONAYLANDI")
            st.caption(f"Model: {selected_model_name} | Risk Olasiligi: %{probability*100:.2f}")
            risk_text = "ONAYLANDI"
        else:
            st.error("KREDİ REDDEDİLDİ")
            st.caption(f"Model: {selected_model_name} | Risk Olasiligi: %{probability*100:.2f}")
            risk_text = "REDDEDILDI"
            
        st.progress(int(probability * 100))

    with c2:
        st.write("Raporlama")
        pdf_bytes = create_pdf(raw_input, prediction, probability, risk_text, selected_model_name)
        st.download_button(
            label="PDF Raporunu Indir",
            data=pdf_bytes,
            file_name=f"risk_raporu.pdf",
            mime="application/pdf"
        )

    if stats_available:
        st.subheader("Analiz Detaylari (Ortalamalarla Kiyaslama)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Gelir", f"{income} TL", f"{income - avg_income:.0f} TL")
        m2.metric("Kredi Tutari", f"{loan_amount} TL", f"{loan_amount - avg_loan_amnt:.0f} TL", delta_color="inverse")
        m3.metric("Faiz", f"%{int_rate}", f"{int_rate - avg_int_rate:.2f}", delta_color="inverse")

st.divider()
st.header("Toplu Analiz (CSV Yukle)")

uploaded_file = st.file_uploader("CSV Dosyasini Surukleyin", type=["csv"])

if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file)
        st.write("Yuklenen Veri (Ilk 5 Satir):")
        st.dataframe(df_upload.head())
        
        if st.button("Toplu Analizi Baslat"):
            df_proc = pd.get_dummies(df_upload)
            df_proc = df_proc.reindex(columns=model_columns, fill_value=0)
            
            preds = current_model.predict(df_proc)
            probs = current_model.predict_proba(df_proc)[:, 1]
            
            df_upload['Tahmin'] = ["RED" if x == 1 else "ONAY" for x in preds]
            df_upload['Risk_Skoru'] = probs
            
            st.success("Islem Tamamlandi")
            st.dataframe(df_upload)
            
            csv_data = df_upload.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Sonuclari Indir (CSV)",
                data=csv_data,
                file_name='toplu_tahmin_sonuclari.csv',
                mime='text/csv',
            )
    except Exception as e:
        st.error(f"Hata: {e}")