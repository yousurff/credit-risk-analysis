# Credit Risk Analysis

This project is a **Machine Learning** model designed to predict credit repayment risk by analyzing the demographic and financial data of bank customers. It includes a comprehensive data analysis notebook and an interactive **Web Interface** for real-time predictions.

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

## Project Overview

Developed within the scope of the Data Mining course, this project aims to classify credit applications as **Approved** or **Denied** (Risk/No Risk) using the **Decision Tree** algorithm.

The project implements theoretical concepts covered in the **Chapter 2 (Data Preprocessing)** and **Chapter 3 (Classification)** lecture notes.

### Objective
To create a decision support system that enables financial institutions to minimize credit default risk based on customer data (e.g., Age, Income, Employment Length, Home Ownership Status).

---

## 📂 Project Structure & File Descriptions

Understanding the key files in this repository:

* **`project.ipynb`**: The main Jupyter Notebook containing the Data Mining pipeline: Data Cleaning, EDA (Exploratory Data Analysis), Visualization, Model Training, and Evaluation.
* **`app.py`**: The source code for the interactive Web Application built with **Streamlit**. It allows users to input new customer data and get a risk prediction instantly.
* **`decision_tree_model.pkl`**: The trained Decision Tree model serialized using `joblib`. This file allows the web application to use the pre-trained AI without retraining it every time.
* **`model_columns.pkl`**: Stores the list of column names used during training. This ensures that the user input in the web app matches the exact format required by the model (especially for One-Hot Encoded features).

---

## Data & Preprocessing

**Dataset Used:** Credit Risk Dataset
* **Input Variables (Features):** Age, Income, Home Ownership, Loan Amount, Interest Rate, Loan Intent, etc.
* **Target Variable:** `loan_status` (0: Non-Default/Paid, 1: Default/Risk)

**Preprocessing Steps applied (based on Chapter 2):**
1.  **Data Cleaning:** Removed outliers (e.g., Age > 100, Employment Length > 100 years).
2.  **Imputation:** Missing values in interest rates and employment length were filled with the mean value.
3.  **Encoding:** Categorical variables like `RENT`, `OWN`, and `EDUCATION` were converted into numerical format using **One-Hot Encoding**.

---

## Modeling

Since the goal is to predict a category, the **Decision Tree Classifier** was selected as the primary model.

* **Train/Test Split:** 80% Training, 20% Testing.
* **Libraries:** Python, Pandas, Scikit-learn, Matplotlib, Seaborn, Streamlit.

---

## Results

Performance on the Test Set:

* **Accuracy:** **88%**
* **Evaluation:** The model is highly effective at identifying non-risky customers with a success rate of over 90%.

**Visualizations:**
* **Confusion Matrix:** A detailed heatmap showing correct and incorrect predictions is included in the project files.
* **Feature Importance:** The model identified the following as the most critical factors for credit approval:
    1.  Customer Income (`person_income`)
    2.  Loan Amount (`loan_amnt`)
    3.  Interest Rate (`loan_int_rate`)

---

## Installation & Usage

To run this project locally (both the notebook and the web app):

### 1. Clone the Repository
```bash
git clone [https://github.com/yousurff/credit-risk-analysis.git](https://github.com/yousurff/credit-risk-analysis.git)
cd credit-risk-analysis
```

### 2. Install Required Libraries

You need streamlit and joblib in addition to standard data science libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

### 3. Run the Web Interface (App)

![Streamlit Arayüzü](screenshot.png)

To start the interactive prediction system:
```bash
streamlit run app.py
```
Note: If you encounter a PATH error, try using: python3 -m streamlit run app.py

### 4. Run the Analysis Notebook

To view the codes, charts, and training process: Open project.ipynb in VS Code or Jupyter Notebook.

---

This project was prepared as the Final Assignment for CEN 416.
