import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

print("⏳ Veri seti yükleniyor ve modeller eğitiliyor... Lütfen bekleyin.")

# 1. Veriyi Yükle ve Temizle
try:
    df = pd.read_csv('credit_risk_dataset.csv')
except FileNotFoundError:
    print("HATA: 'credit_risk_dataset.csv' dosyası bulunamadı! Lütfen proje klasöründe olduğundan emin ol.")
    exit()

# Temizlik (Notebook'taki adımların aynısı)
df = df[df['person_age'] < 100]
df = df[df['person_emp_length'] < 100]
df['person_emp_length'].fillna(df['person_emp_length'].mean(), inplace=True)
df['loan_int_rate'].fillna(df['loan_int_rate'].mean(), inplace=True)

# Kategorik Dönüşüm
cat_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
df = pd.get_dummies(df, columns=cat_columns, drop_first=True)

# X ve y ayır
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Eğitim seti
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODELLERİ EĞİT VE KAYDET ---

# 1. Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
joblib.dump(dt_model, 'decision_tree_model.pkl')
print("✅ Decision Tree kaydedildi.")

# 2. Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'random_forest_model.pkl')
print("✅ Random Forest kaydedildi.")

# 3. XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, 'xgboost_model.pkl')
print("✅ XGBoost kaydedildi.")

# 4. Logistic Regression
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)
joblib.dump(log_model, 'logistic_model.pkl')
print("✅ Logistic Regression kaydedildi.")

# 5. Sütun İsimlerini Kaydet (Çok Önemli!)
joblib.dump(X.columns, 'model_columns.pkl')
print("✅ Model sütunları kaydedildi.")

print("\n🎉 İŞLEM TAMAM! Şimdi 'streamlit run app.py' komutunu çalıştırabilirsin.")