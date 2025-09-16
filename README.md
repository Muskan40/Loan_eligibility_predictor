
# 🏦 Loan Eligibility Prediction System  

An end-to-end **machine learning project** that predicts whether a customer is eligible for a loan, based on their profile information. This project includes **data preprocessing, model training, evaluation, and deployment as a Streamlit web app**.  

🔗 **Live Demo:** [Streamlit App Link](https://loaneligibilitypredictor-5stpztxkkhmvbfzovtnwoe.streamlit.app/)  

---

## 📖 Problem Statement  

Dream Housing Finance Company wants to automate the **loan eligibility process** based on customer details provided in an online application form. The dataset includes features like:  
- Gender  
- Marital Status  
- Education  
- Dependents  
- Applicant Income  
- Loan Amount  
- Credit History  
- Property Area  

The goal is to **predict loan eligibility** (`Yes` / `No`) and deploy the model for real-time use.  

---

## 📊 Dataset  

- Source: [Analytics Vidhya Loan Prediction Dataset](https://www.analyticsvidhya.com/datahack/contest/practice-problem-loan-prediction-iii/)  
- Train set: `train_ctrUa4K.csv`  
- Test set: `test_lAUu6dG.csv`  

---

## ⚙️ Tech Stack  

- **Python** (Pandas, NumPy, Matplotlib, Seaborn)  
- **Scikit-learn** (Logistic Regression, Random Forest, preprocessing utilities)  
- **XGBoost** (final model of choice)  
- **imbalanced-learn** (SMOTE for class imbalance)  
- **Streamlit** (deployment)  

---

## 🛠️ Workflow  

1. **Data Exploration & EDA**  
   - Distribution analysis of categorical & numerical features  
   - Visualizations: countplots, histograms, violin plots, heatmaps  

2. **Data Preprocessing**  
   - Handling missing values  
   - Encoding categorical variables with one-hot encoding  
   - Outlier detection and removal (IQR method)  
   - Skewness correction using transformations  
   - Class imbalance handled with **SMOTE**  
   - Feature scaling with **MinMaxScaler**  

3. **Modeling**  
   - Models trained & compared:  
     - Logistic Regression  
     - Random Forest Classifier  
     - XGBoost Classifier  
   - Evaluation metrics: Accuracy, ROC-AUC, Confusion Matrix, Classification Report  

4. **Model Insights**  
   - Property area and marital status showed stronger influence than income or credit history.  
   - Gender and education were less impactful.  
   - **XGBoost delivered the best performance** and was chosen for deployment.  

5. **Deployment**  
   - Built an interactive **Streamlit app** for real-time predictions.  
   - Users can enter loan applicant details and get instant eligibility results.  

---

## 🚀 How to Run Locally  

Clone the repo:  
```bash
git clone https://github.com/yourusername/loan-eligibility-predictor.git
cd loan-eligibility-predictor
````

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📂 Repository Structure

```
📦 loan-eligibility-predictor
 ┣ 📜 app.py                # Streamlit app
 ┣ 📜 notebook.ipynb        # Jupyter notebook with full analysis & modeling
 ┣ 📜 train_ctrUa4K.csv     # Training dataset
 ┣ 📜 test_lAUu6dG.csv      # Test dataset
 ┣ 📜 requirements.txt      # Dependencies
 ┣ 📜 LoanPredictions.csv   # Sample predictions
 ┣ 📜 README.md             # Project documentation
```

---

## 📈 Results

* Logistic Regression: Moderate performance, baseline model
* Random Forest: Improved performance with feature importance insights
* **XGBoost: Best-performing model, selected for deployment**

---

## 🌐 Live Demo

👉 [Streamlit App Link](https://loaneligibilitypredictor-5stpztxkkhmvbfzovtnwoe.streamlit.app/)

---

## 📌 Future Improvements

* Hyperparameter tuning with GridSearchCV / Optuna
* Deployment with Docker & cloud hosting (AWS / GCP / Azure)
* Building an API with FastAPI/Flask for wider integration
* Improving explainability with SHAP or LIME

---

## 🤝 Contributing

Contributions are welcome! Please fork the repo and submit a pull request.

---

