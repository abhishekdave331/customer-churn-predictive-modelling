# 📉 Customer Churn Prediction – End-to-End Case Study

This project aims to predict customer churn using various machine learning models based on a structured dataset from a banking domain. The complete pipeline includes data exploration, preprocessing, feature engineering, model training, evaluation, and deployment preparation.

---

## 📌 Problem Statement

Churn prediction helps businesses identify customers likely to leave their services. By analyzing customer data, we aim to build a classification model that can predict whether a customer will churn, enabling timely retention strategies.

---

## 🧰 Tools & Technologies Used

- **Python**
- **Pandas, NumPy** – Data manipulation
- **Matplotlib, Seaborn** – Visualization
- **Scikit-learn** – ML modeling and preprocessing
- **XGBoost** – Advanced ensemble learning
- **Pickle** – Model saving for deployment

---

## 🔍 Project Workflow

### 1. 📦 Data Loading & Exploration
- Loaded `Churn_Modelling.csv` dataset
- Basic EDA: `.info()`, `.describe()`, missing values check
- Correlation analysis among numerical features

### 2. 📊 Exploratory Data Analysis (EDA)
- Univariate, bivariate, and multivariate visualizations
- KDE plots for numerical variables
- Categorical variable distribution

### 3. 🧹 Data Preprocessing
- Handled missing values and outliers
- Categorical encoding using LabelEncoder
- Feature scaling with MinMaxScaler, StandardScaler, and RobustScaler
- Applied Power Transformation for skewed distributions

### 4. 🏗️ Feature Engineering
- Selection of relevant features
- Transformation for better model performance

### 5. 🤖 Model Building
Implemented and compared the following models:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost

### 6. ✅ Model Evaluation
- Evaluated with metrics: Accuracy, Precision, Recall, F1-score
- Compared model performance and selected best one

### 7. 💾 Model Saving
- Serialized the final model using `pickle` for future deployment

---

## 📈 Key Results

- Achieved high accuracy and generalization using ensemble models
- Identified key features influencing churn (e.g., Age, Balance, IsActiveMember)

---

## 📂 Folder Structure

📦customer-churn-case-study
┣ 📜Churn_Modelling.csv
┣ 📜Customer_Churn_Case_Study.ipynb
┣ 📜README.md
┗ 📜model.pkl

---

## 🚀 Future Improvements

- Implement cross-validation
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Integrate with a web app (e.g., Flask or Streamlit)
- Business dashboard with Power BI or Tableau

---

## 📬 Contact

For questions or collaboration:  
📧 abhishekdave331@gmail.com 
🔗 [LinkedIn](www.linkedin.com/in/abhishek-dave-)

---

