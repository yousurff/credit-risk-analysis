# Credit Risk Analysis System

This project is a comprehensive **Machine Learning** solution designed to predict credit repayment risk. Unlike standard classification models, this system includes an **Interactive Dashboard** that provides **statistical reasoning** for rejections and supports **Batch Processing** for financial institutions.

## Course & Student Information

| Detail | Information |
|--------|-------------|
| **University** | Maltepe University |
| **Department** | Software Engineering (3rd Year) |
| **Course Code** | CEN 416 01 - Data Mining |
| **Instructor** | Asst. Prof. Dr. Önder TOMBUŞ |
| **Student Name** | Yusuf Talha KAMİLOĞLU |
| **Student ID** | 220706006 |

---

## Key Features (Why this project is unique?)

This project goes beyond simple prediction by implementing **Explainable AI (XAI)** concepts and practical software engineering principles:

1.  **🤖 AI-Powered Risk Prediction:** Uses a trained **Decision Tree Classifier** to assess creditworthiness instantly.
2.  **📊 Comparative Statistical Analysis:** It doesn't just say "No". It explains **why**. The system compares the applicant's data (Income, Loan Amount, Interest Rate) with the average of *approved customers*, highlighting the specific reasons for rejection.
3.  **📂 Batch Processing (CSV Upload):** Designed for real-world banking scenarios. Users can upload a CSV file containing thousands of applicants, and the system processes them in bulk, exporting the results as a downloadable file.
4.  **📉 Real-time Visualization:** Dynamic charts and metrics (Green/Red indicators) provide immediate visual feedback.

---

## Project Structure

* **\`project.ipynb\`**: The Data Mining pipeline (Data Cleaning, Outlier Removal, EDA, Model Training).
* **\`app.py\`**: The source code for the **Streamlit Web Application**.
* **\`decision_tree_model.pkl\`**: The serialized Machine Learning model.
* **\`model_columns.pkl\`**: Ensures exact feature matching between the training set and the web app inputs.
* **\`credit_risk_dataset.csv\`**: The dataset used for training and calculating statistical averages.

---

## Data & Methodology

**Dataset:** Credit Risk Dataset (Source: Kaggle)
* **Target:** \`loan_status\` (0: Approved, 1: Risk/Default)
* **Key Features:** Age, Income, Employment Length, Loan Amount, Interest Rate, Home Ownership, etc.

**Preprocessing (Chapter 2):**
* Outlier Detection (Removed unrealistic ages > 100).
* Imputation (Filled missing employment years and interest rates).
* One-Hot Encoding (Converted categorical data to numeric).

**Modeling (Chapter 3):**
* **Algorithm:** Decision Tree Classifier.
* **Performance:** **88% Accuracy** on the Test Set.
* **Validation:** 80-20 Train-Test Split validation.

---

## Installation & Usage

To run the full dashboard locally:

### 1. Clone the Repository
\`\`\`bash
git clone https://github.com/yousurff/credit-risk-analysis.git
cd credit-risk-analysis
\`\`\`

### 2. Install Dependencies
\`\`\`bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
\`\`\`

### 3. Run the Application
\`\`\`bash
streamlit run app.py
\`\`\`
*The web interface will open automatically in your browser.*
![Streamlit Arayüzü](screenshot.png)

---
*This project was prepared as the Final Assignment for CEN 416.*
