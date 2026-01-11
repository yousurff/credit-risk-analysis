import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. AYARLAR VE MODEL YÜKLEME ---
st.set_page_config(page_title="Kredi Risk Analizi", page_icon="🏦", layout="wide")

try:
    model = joblib.load('decision_tree_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
except:
    st.error("Model dosyaları (pkl) bulunamadı! Lütfen önce proje.ipynb dosyasındaki kaydetme kodunu çalıştırın.")
    st.stop()

# İstatistikler için orijinal veriyi yüklemeye çalışalım
# Bu, ortalamaları hesaplamak için gerekli.
try:
    df_orj = pd.read_csv('credit_risk_dataset.csv')
    # Basit temizlik (Outlier temizliği - Projedeki gibi)
    df_orj = df_orj[df_orj['person_age'] < 100]
    df_orj = df_orj[df_orj['person_emp_length'] < 100]
    
    # Sadece Kredisi ONAYLANANLARIN (loan_status=0) ortalamalarını al
    df_approved = df_orj[df_orj['loan_status'] == 0]
    
    avg_income = df_approved['person_income'].mean()
    avg_loan_amnt = df_approved['loan_amnt'].mean()
    avg_emp_length = df_approved['person_emp_length'].mean()
    avg_int_rate = df_approved['loan_int_rate'].mean()
    
    stats_available = True
except:
    stats_available = False
    st.warning("⚠️ 'credit_risk_dataset.csv' dosyası bulunamadığı için karşılaştırmalı istatistikler gösterilemiyor.")

# --- 2. ARAYÜZ BAŞLIĞI ---
st.title("Kredi Risk Tahmin Sistemi")
st.markdown("**Maltepe Üniversitesi - Yazılım Müh. - CEN 416 Final Projesi**")
st.markdown("---")

# --- 3. SOL MENÜ (GİRDİLER) ---
st.sidebar.header("Müşteri Bilgileri")

age = st.sidebar.number_input("Yaş", min_value=18, max_value=100, value=25)
income = st.sidebar.number_input("Yıllık Gelir (TL)", min_value=0, value=50000, step=1000)
emp_length = st.sidebar.slider("Çalışma Süresi (Yıl)", 0.0, 40.0, 5.0)
loan_amount = st.sidebar.number_input("İstenen Kredi Miktarı", min_value=0, value=10000, step=500)
int_rate = st.sidebar.number_input("Faiz Oranı (%)", min_value=0.0, value=10.0, step=0.1)

home_ownership = st.sidebar.selectbox("Ev Durumu", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
loan_intent = st.sidebar.selectbox("Kredi Amacı", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
loan_grade = st.sidebar.selectbox("Kredi Derecesi", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
default_history = st.sidebar.selectbox("Daha önce temerrüde düştü mü?", ['Y', 'N'])

# --- 4. TAHMİN İŞLEMİ ---
if st.button("Risk Analizi Yap", type="primary"):
    
    # Veriyi DataFrame'e çevir
    input_data = pd.DataFrame({
        'person_age': [age],
        'person_income': [income],
        'person_emp_length': [emp_length],
        'loan_amnt': [loan_amount],
        'loan_int_rate': [int_rate],
        'loan_percent_income': [loan_amount / income if income > 0 else 0],
        'cb_person_cred_hist_length': [2], # Varsayılan
        'person_home_ownership': [home_ownership],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'cb_person_default_on_file': [default_history]
    })

    # One-Hot Encoding ve Reindex
    input_dummies = pd.get_dummies(input_data)
    input_final = input_dummies.reindex(columns=model_columns, fill_value=0)

    # Tahmin
    prediction = model.predict(input_final)
    probability = model.predict_proba(input_final)[0][1] 

    # --- 5. SONUÇ GÖSTERİMİ ---
    st.divider()
    
    col_res1, col_res2 = st.columns([2, 1])
    
    with col_res1:
        if prediction[0] == 0:
            st.success("**SONUÇ: KREDİ ONAYLANDI (Düşük Risk)**")
            st.balloons()
        else:
            st.error("**SONUÇ: KREDİ REDDEDİLDİ (Yüksek Risk)**")
            st.markdown(f"**Risk Skoru:** %{probability*100:.2f}")

    # --- 6. NEDEN REDDEDİLDİ? / ORTALAMA KIYASLAMASI ---
    if stats_available:
        st.subheader("📊 Neden? - Kredisi Onaylananlar ile Karşılaştırma")
        st.write("Aşağıdaki oklar, kredisi onaylanan ortalama bir müşteriye göre durumunuzu gösterir.")
        
        m1, m2, m3, m4 = st.columns(4)
        
        # Gelir Kıyaslaması (Yüksek olması iyi - Yeşil)
        m1.metric(
            label="Yıllık Gelir", 
            value=f"{income:,.0f} TL", 
            delta=f"{income - avg_income:,.0f} TL",
            delta_color="normal" # Artı ise yeşil (iyi)
        )
        
        # Kredi Miktarı Kıyaslaması (Düşük olması iyi - delta color inverse)
        m2.metric(
            label="Kredi Miktarı", 
            value=f"{loan_amount:,.0f} TL", 
            delta=f"{loan_amount - avg_loan_amnt:,.0f} TL",
            delta_color="inverse" # Eksi ise yeşil (çünkü az borç iyidir)
        )
        
        # Çalışma Süresi (Yüksek olması iyi)
        m3.metric(
            label="Çalışma Süresi", 
            value=f"{emp_length} Yıl", 
            delta=f"{emp_length - avg_emp_length:.1f} Yıl",
            delta_color="normal"
        )
        
        # Faiz Oranı (Düşük olması iyi)
        m4.metric(
            label="Faiz Oranı", 
            value=f"%{int_rate}", 
            delta=f"{int_rate - avg_int_rate:.2f}",
            delta_color="inverse"
        )
        
        st.caption(f"*Not: Kredisi onaylananların ortalama geliri {avg_income:,.0f} TL ve talep ettikleri ortalama kredi {avg_loan_amnt:,.0f} TL'dir.*")

# --- 7. TOPLU ANALİZ (CSV YÜKLEME) ---
st.markdown("---")
st.header("Toplu Analiz (Excel/CSV Yükle)")
st.info("Elinizde birden fazla müşteri varsa, CSV dosyasını buraya yükleyerek toplu tahmin alabilirsiniz.")

uploaded_file = st.file_uploader("CSV Dosyasını Sürükleyin", type=["csv"])

if uploaded_file is not None:
    try:
        # Dosyayı oku
        df_upload = pd.read_csv(uploaded_file)
        st.write("Yüklenen Veri (İlk 5 Satır):")
        st.dataframe(df_upload.head())
        
        if st.button("Toplu Analizi Başlat"):
            with st.spinner('Yapay Zeka düşünüyor...'):
                # İşleme
                df_proc = pd.get_dummies(df_upload)
                df_proc = df_proc.reindex(columns=model_columns, fill_value=0)
                
                # Tahmin
                preds = model.predict(df_proc)
                probs = model.predict_proba(df_proc)[:, 1]
                
                # Sonuçları ekle
                df_upload['Tahmin'] = ["RED" if x == 1 else "ONAY" for x in preds]
                df_upload['Risk_Skoru'] = probs
                
                st.success("İşlem Tamamlandı!")
                st.dataframe(df_upload)
                
                # İndirme Butonu
                csv_data = df_upload.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Sonuçları İndir (CSV)",
                    data=csv_data,
                    file_name='kredi_tahmin_sonuclari.csv',
                    mime='text/csv',
                )
    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")