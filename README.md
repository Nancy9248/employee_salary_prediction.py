# employee_salary_prediction.py
Machine Learning project to predict employee salary category (&lt;=50K or >50K) using the Adult Income dataset and a Streamlit web app.
# Employee Salary Prediction

This project predicts whether an employee earns **more than $50K per year** based on demographic and work-related attributes, using machine learning models trained on the Adult Income dataset.

---

## Project Overview

**Objective:**
To build a classifier that predicts income category (`<=50K` or `>50K`) based on factors like:
- Age
- Education
- Occupation
- Hours worked per week
- Marital status
- and more.

This can help HR departments and analysts understand salary trends and make data-driven decisions.

---

## ⚙️ Project Components

 **employee_salary_prediction.py**  
Cleans data, trains a Random Forest model, and evaluates performance.

 **streamlit_salary_predictor.py**  
A Streamlit web app where users input details to get a salary prediction.

---

## Dataset

This project uses the **Adult Income Dataset** (also known as the "Census Income" dataset).  
File used: `adult 3.csv`

---

## How to Run

## 1. Python Script (Model Training)

```bash
pip install pandas scikit-learn
python employee_salary_prediction.py


pip install streamlit
streamlit run streamlit_salary_predictor.py

Accuracy: 0.85
Classification Report: ...

