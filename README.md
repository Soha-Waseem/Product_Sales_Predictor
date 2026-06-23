# Advertising Sales Prediction using Polynomial Regression

## Project Overview

This project uses Machine Learning to predict product sales based on advertising budgets spent on **TV, Radio, and Newspaper**.

A **Polynomial Regression model** is used to capture non-linear relationships between advertising spending and sales. The project also includes a **Streamlit web application** that allows users to enter advertising budgets and get real-time sales predictions.

---

## Features

- Data loading and preprocessing
- Feature selection from advertising dataset
- Polynomial Regression model training
- Captures non-linear relationships
- Predicts product sales from advertising budgets
- Model evaluation using:
  - R² Score
- Visualization of advertising impact on sales
- Interactive Streamlit web interface
- User input-based sales prediction

---

## Machine Learning Model

### Algorithm:
**Polynomial Regression**

### Workflow:

1. Load advertising dataset
2. Select input features:
   - TV
   - Radio
   - Newspaper
3. Apply Polynomial Feature Transformation
4. Train Linear Regression model
5. Evaluate model performance
6. Predict sales for new advertising budgets

Polynomial Regression helps identify patterns like **diminishing returns**, where increasing advertising spending may produce smaller increases in sales.

---

## Dataset

* 200 data points (`advertising.csv`)
* Inputs:

  * TV advertising spend
  * Radio advertising spend
  * Newspaper advertising spend
* Output:

  * Product Sales 

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

---

##  Project Structure

* `app.py` → Streamlit web application
* `advertising.csv` → Dataset
* `Polynomial_Regression_model.ipynb` → Model training and testing
* `requirements.txt` → Required files

##  How to Run the Project

### 1. Install Required Libraries
* pip install -r requirements.txt

### 2. Run the Machine Learning Model
* python Polynomial_Regression_model.py

### 3. Run the Streamlit Web Application
* streamlit run app.py
  
