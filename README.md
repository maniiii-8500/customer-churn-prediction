# Customer Churn Prediction System

A machine learning application that predicts whether a telecom customer is likely to churn based on customer demographics, services, contract details, and billing information.

## Features

- Exploratory Data Analysis using Pandas, NumPy, Matplotlib, and Seaborn
- Data preprocessing and categorical feature encoding
- Comparison of Logistic Regression, Decision Tree, and Random Forest models
- Customer churn probability prediction
- Interactive Streamlit web application
- Saved machine learning pipeline for deployment

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Project Structure

customer-churn-prediction/
├── app/
│   └── app.py
├── data/
│   └── customer_churn.csv
├── models/
│   └── customer_churn_pipeline.pkl
├── notebooks/
│   └── customer_churn_analysis.ipynb
├── src/
├── requirements.txt
└── .gitignore

## Model Performance

Random Forest was selected as the final model based on its overall performance for churn prediction.

Key evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Run Locally

1. Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL

2. Create and activate a virtual environment:

python -m venv venv
venv\Scripts\activate

3. Install dependencies:

pip install -r requirements.txt

4. Run the Streamlit application:

streamlit run app/app.py

The application will open in your browser at:

http://localhost:8501

## Author

Nagamani Dasari
