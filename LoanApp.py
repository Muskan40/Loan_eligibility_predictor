# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import missingno as mso
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Loan Prediction — Streamlit App", layout="wide")

# -------------------------
# Default training file (from your notebook)
# -------------------------
TRAIN_PATH = "train_ctrUa4K.csv"  # <-- exact filename from your notebook

# Sidebar: files & settings
if "log_reg" not in st.session_state:
    st.session_state.log_reg = None
if "rf" not in st.session_state:
    st.session_state.rf = None
if "xgb" not in st.session_state:
    st.session_state.xgb = None # placeholders for trained models
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols=None

with st.sidebar:
    st.title("Data & Files")
    st.markdown("**About dataset**")
    st.caption(
        "Dream Housing Finance — Loan prediction dataset (Analytics Vidhya practice)."
    )
    st.markdown(
        "[Dataset source — Analytics Vidhya practice problem](https://www.analyticsvidhya.com/datahack/contest/practice-problem-loan-prediction-iii/)"
    )
    use_default = st.checkbox("Use default training file (notebook)", value=True)
    uploaded_train = st.file_uploader("Upload training file (CSV/XLSX)", type=["csv", "xlsx"])
    uploaded_test = st.file_uploader("Upload testing file (optional)", type=["csv", "xlsx"])
    preview_rows = st.slider("Preview rows", min_value=5, max_value=200, value=10)
    st.markdown("---")
    st.markdown("**Quick controls**")
    random_seed = st.number_input("Random seed", value=42, step=1)
    st.caption("Defaults will follow your notebook unless you upload files here.")

# -------------------------
# Utility: load dataframe
# -------------------------
@st.cache_data(show_spinner=False)
def load_df(path=None, uploaded_file=None):
    if uploaded_file is not None:
        # UploadedFile is an io.BytesIO-like object
        try:
            name = getattr(uploaded_file, "name", "")
            if name.endswith(".csv") or str(name).lower().endswith(".csv"):
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file)
            else:
                uploaded_file.seek(0)
                return pd.read_excel(uploaded_file)
        except Exception:
            # Fallback: try csv
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)
    if path:
        return pd.read_csv(path)
    return None

# choose which source to load
train_path = TRAIN_PATH if use_default and uploaded_train is None else None
df = load_df(path=train_path, uploaded_file=uploaded_train)

# store df in session_state for later tabs
if df is not None:
    st.session_state["train_df"] = df

# -------------------------
# Top: Title & short description
# -------------------------
st.title("Loan Eligibility — Interactive App")
st.markdown(
    """
This app is a Streamlit conversion of your notebook for the **Loan Prediction** problem.
- Default training dataset: `train_ctrUa4K.csv` (from your notebook).
- You can upload alternate training or testing files from the sidebar.
"""
)

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(
    [
        "Dataset preview",
        "EDA",
        "Model training",
        "Predictions",
        "Single prediction",
        "Interpretability",
        "Insights",
    ]
)

# -------------------------
# 1) Dataset preview
# -------------------------
with tabs[0]:
    st.header("Dataset preview")
    st.subheader("About the dataset")
    st.markdown(
        """
- **Problem statement:** Automate loan eligibility based on customer details (Gender, Marital Status, Education,
  Dependents, Income, Loan Amount, Credit History, etc.).
- **Source:** Analytics Vidhya practice problem (link in sidebar).
"""
    )

    if df is None:
        st.warning(
            "No training data loaded. Either enable 'Use default training file' in the sidebar or upload a training file."
        )
    else:
        st.write("**Shape:**", df.shape)
        st.dataframe(df.head(preview_rows))
        col1, col2 = st.columns(2)
        with col1:
            if st.checkbox("Show dtypes"):
                st.write(df.dtypes)
        with col2:
            if st.checkbox("Show missing value counts"):
                mv = df.isnull().sum().sort_values(ascending=False)
                st.write(mv[mv > 0])

        if st.checkbox("Show basic descriptive stats"):
            st.write(df.describe().T)

# -------------------------
# 2) EDA (placeholder)
# -------------------------
with tabs[1]:
    st.header(" Data Preprocessing & EDA")
    st.subheader("Exploratory Data Analysis (EDA)")
    if "train_df" not in st.session_state:
        st.warning("No dataset loaded.")
    else:
        edf = st.session_state["train_df"]

        st.subheader("Categorical Variables")
        cat_options = [
            "Gender",
            "Married",
            "Education",
            "Self_Employed",
            "Credit_History",
            "Loan_Amount_Term",
            "Property_Area",
            "Loan_Status",
        ]
        cat_choice = st.selectbox("Select a categorical variable", cat_options)

        if cat_choice:
            st.write(edf[cat_choice].value_counts(dropna=False))
            fig, ax = plt.subplots()
            sns.countplot(x=cat_choice, data=edf, palette="hls", ax=ax)
            st.pyplot(fig)

        st.subheader("Numerical Variables")
        if st.checkbox("Show histograms"):
            num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            sns.histplot(x="ApplicantIncome", data=edf, ax=axs[0, 0], kde=True, color="green")
            sns.histplot(x="CoapplicantIncome", data=edf, ax=axs[0, 1], kde=True, color="red")
            sns.histplot(x="LoanAmount", data=edf, ax=axs[1, 0], kde=True, color="blue")
            sns.histplot(x="Loan_Amount_Term", data=edf, ax=axs[1, 1], kde=True, color="orange")
            st.pyplot(fig)

        if st.checkbox("Show violin plots"):
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            sns.violinplot(y="ApplicantIncome", data=edf, ax=axs[0, 0], color="green")
            sns.violinplot(y="CoapplicantIncome", data=edf, ax=axs[0, 1], color="red")
            sns.violinplot(y="LoanAmount", data=edf, ax=axs[1, 0], color="blue")
            sns.violinplot(y="Loan_Amount_Term", data=edf, ax=axs[1, 1], color="orange")
            st.pyplot(fig)

        st.subheader("Bivariate Analysis")
        if st.checkbox("Correlation Heatmap"):
            corr_matrix = edf.corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        if st.checkbox("Crosstabs & Barplots"):
            fig1 = pd.crosstab(edf.Gender, edf.Married).plot(kind="bar", figsize=(6, 4)).get_figure()
            st.pyplot(fig1)

            fig2 = pd.crosstab(edf.Self_Employed, edf.Credit_History).plot(kind="bar", figsize=(6, 4)).get_figure()
            st.pyplot(fig2)

            fig3 = pd.crosstab(edf.Self_Employed, edf.Loan_Status).plot(kind="bar", figsize=(6, 4)).get_figure()
            st.pyplot(fig3)

            fig4 = pd.crosstab(edf.Property_Area, edf.Loan_Status).plot(kind="bar", figsize=(6, 4)).get_figure()
            st.pyplot(fig4)

        if st.checkbox("Boxplots vs Loan_Status"):
            fig, axs = plt.subplots(1, 3, figsize=(14, 4))
            sns.boxplot(y="LoanAmount", x="Loan_Status", data=edf, ax=axs[0], palette="mako")
            sns.boxplot(y="ApplicantIncome", x="Loan_Status", data=edf, ax=axs[1], palette="mako")
            sns.boxplot(y="CoapplicantIncome", x="Loan_Status", data=edf, ax=axs[2], palette="mako")
            st.pyplot(fig)

        if st.checkbox("Applicant vs Coapplicant Scatter"):
            fig, ax = plt.subplots()
            edf.plot(x="CoapplicantIncome", y="ApplicantIncome", style="o", ax=ax)
            st.pyplot(fig)

        st.subheader("Missing Values")
        if st.checkbox("Show missingno bar chart"):
            fig = mso.bar(edf)
            st.pyplot(fig.figure)
            
    st.subheader(" Handling Missing Values & Cleaning")
    df_proc = df.drop(['Loan_ID'], axis=1)

    # Fill missing values
    df_proc['Gender'].fillna(df_proc['Gender'].mode()[0], inplace=True)
    df_proc['Married'].fillna(df_proc['Married'].mode()[0], inplace=True)
    df_proc['Dependents'].fillna(df_proc['Dependents'].mode()[0], inplace=True)
    df_proc['Self_Employed'].fillna(df_proc['Self_Employed'].mode()[0], inplace=True)
    df_proc['Credit_History'].fillna(df_proc['Credit_History'].mode()[0], inplace=True)
    df_proc['Loan_Amount_Term'].fillna(df_proc['Loan_Amount_Term'].mode()[0], inplace=True)
    df_proc['LoanAmount'].fillna(df_proc['LoanAmount'].mean(), inplace=True)

    st.write("Remaining Null Values after Imputation:")
    st.write(df_proc.isnull().sum())

    # One-hot encoding
    df_proc = pd.get_dummies(df_proc)
    df_proc = df_proc.drop(['Gender_Female','Married_No','Education_Not Graduate',
                            'Self_Employed_No','Loan_Status_N'], axis=1)

    for col in df_proc.columns:
        if df_proc[col].dtype == 'bool':
            df_proc[col] = df_proc[col].astype(np.uint8)

    df_proc.rename(columns={
        'Married_Yes':'Married',
        'Self_Employed_Yes':'Self_Employed',
        'Loan_Status_Y':'Loan_Status'
    }, inplace=True)

    st.subheader("Outlier Removal (IQR Method)")
    Q1 = df_proc.quantile(0.25)
    Q3 = df_proc.quantile(0.75)
    IQR = Q3 - Q1
    df_proc = df_proc[~((df_proc < (Q1 - 1.5 * IQR)) | 
                        (df_proc > (Q3 + 1.5 * IQR))).any(axis=1)]
    st.write("Shape after removing outliers:", df_proc.shape)

    st.subheader("Feature Distributions")
    fig, axs = plt.subplots(2,2, figsize=(10,8))
    sns.histplot(x='ApplicantIncome', data=df_proc, ax=axs[0,0], kde=True, color='green')
    sns.histplot(x='CoapplicantIncome', data=df_proc, ax=axs[0,1], kde=True, color='red')
    sns.histplot(x='LoanAmount', data=df_proc, ax=axs[1,0], kde=True, color='blue')
    st.pyplot(fig)

    # Transform skewed features
    df_proc['ApplicantIncome'] = np.sqrt(df_proc['ApplicantIncome'])
    df_proc['LoanAmount'] = np.sqrt(df_proc['LoanAmount'])
    df_proc['CoapplicantIncome'] = np.sqrt(df_proc['CoapplicantIncome'])

    st.subheader("Loan Status Distribution (Before SMOTE)")
    fig, ax = plt.subplots()
    sns.countplot(y=df_proc['Loan_Status'], palette="coolwarm", ax=ax)
    st.pyplot(fig)

    # Train-test prep
    X = df_proc.drop(['Loan_Status'], axis=1)
    y = df_proc['Loan_Status']

    # Apply SMOTE
    smote = SMOTE()
    X, y = smote.fit_resample(X, y)

    st.subheader("Loan Status Distribution (After SMOTE)")
    fig, ax = plt.subplots()
    sns.countplot(y=y, palette="coolwarm", ax=ax)
    st.pyplot(fig)

    # Scale features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    st.success("✅ Preprocessing Completed! Data is ready for model training.")
    st.write("Final X shape:", X.shape)
    #st.write("Final y distribution:", np.bincount(y))
# -------------------------
# 3) Model training (placeholder)
# -------------------------
with tabs[2]:
    st.header("🤖 Model Training & Evaluation")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=45
    )

    st.write("Shape of X_train:", X_train.shape)
    st.write("Shape of X_test:", X_test.shape)
    st.write("Shape of y_train:", y_train.shape)
    st.write("Shape of y_test:", y_test.shape)

    # Helper function
    def evaluate_model(model, X_test, y_test, name="Model"):
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        st.subheader(f"📌 Results for {name}")
        st.write("**Accuracy:**", acc)
        st.write("**ROC-AUC:**", auc)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Classification Report
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        return acc, auc

    st.subheader("Select Model(s) to Train")
    models = st.multiselect(
        "Choose Models",
        ["Logistic Regression", "Random Forest", "XGBoost"],
        default=["Logistic Regression"]
    )

    if st.button("🚀 Train Selected Models"):
        results = {}
        if "Logistic Regression" in models:
            log_reg = LogisticRegression(max_iter=1000, random_state=42)
            log_reg.fit(X_train, y_train)
            results["Logistic Regression"] = evaluate_model(log_reg, X_test, y_test, "Logistic Regression")
            st.session_state.log_reg = log_reg
        if "Random Forest" in models:
            rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
            rf.fit(X_train, y_train)
            results["Random Forest"] = evaluate_model(rf, X_test, y_test, "Random Forest")
            st.session_state.rf = rf
        if "XGBoost" in models:
            xgb = XGBClassifier(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss"
            )
            xgb.fit(X_train, y_train)
            results["XGBoost"] = evaluate_model(xgb, X_test, y_test, "XGBoost")
            st.session_state.xgb = xgb
        st.success("✅ Model Training Completed!")

# -------------------------
# 4) Predictions (placeholder)
# -------------------------
with tabs[3]:
    st.header("📊 Model Testing & Predictions")

    # Upload Test File
    test_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"], key="test")

    if test_file is not None:
        test_df = pd.read_csv(test_file)
        st.write("Test Data Shape:", test_df.shape)
        st.dataframe(test_df.head())

        loan_ids = test_df["Loan_ID"]   # save for submission later
        test_df = test_df.drop(["Loan_ID"], axis=1)

        # Handle missing values
        test_df['Gender'].fillna(test_df['Gender'].mode()[0], inplace=True)
        test_df['Married'].fillna(test_df['Married'].mode()[0], inplace=True)
        test_df['Dependents'].fillna(test_df['Dependents'].mode()[0], inplace=True)
        test_df['Self_Employed'].fillna(test_df['Self_Employed'].mode()[0], inplace=True)
        test_df['Credit_History'].fillna(test_df['Credit_History'].mode()[0], inplace=True)
        test_df['Loan_Amount_Term'].fillna(test_df['Loan_Amount_Term'].mode()[0], inplace=True)
        test_df['LoanAmount'].fillna(test_df['LoanAmount'].mean(), inplace=True)

        # Encode categorical variables
        test_df = pd.get_dummies(test_df)
        drop_cols = ['Gender_Female','Married_No','Education_Not Graduate','Self_Employed_No']
        test_df = test_df.drop([c for c in drop_cols if c in test_df.columns], axis=1)
        test_df.rename(columns={
            'Married_Yes':'Married',
            'Self_Employed_Yes':'Self_Employed'
        }, inplace=True)

        # Align with training feature columns
        st.session_state.feature_cols = df_proc.drop("Loan_Status", axis=1).columns
        missing_cols = set(st.session_state.feature_cols) - set(test_df.columns)
        for c in missing_cols:
            test_df[c] = 0
        test_df = test_df[st.session_state.feature_cols]

        # Scale test data (same scaler as training)
        scaler = MinMaxScaler()
        scaler.fit(df_proc.drop("Loan_Status", axis=1))
        test_scaled = scaler.transform(test_df)

        # Pick trained models
        try:
            preds_dict = {"Loan_ID": loan_ids}

            # Logistic Regression
            if "log_reg" in st.session_state and st.session_state.log_reg is not None:
                preds_dict["LogReg_Pred"] = st.session_state.log_reg.predict(test_scaled)

            # Random Forest
            if "rf" in st.session_state and st.session_state.rf is not None:
                preds_dict["RF_Pred"] = st.session_state.rf.predict(test_scaled)
                preds_dict["RF_Prob"] = st.session_state.rf.predict_proba(test_scaled)[:, 1]

            # XGBoost
            if "xgb" in st.session_state and st.session_state.xgb is not None:
                preds_dict["XGB_Pred"] = st.session_state.xgb.predict(test_scaled)
                preds_dict["XGB_Prob"] = st.session_state.xgb.predict_proba(test_scaled)[:, 1]

            # Build submission DataFrame
            submission = pd.DataFrame(preds_dict)

            st.subheader("✅ Predictions")
            st.dataframe(submission.head())

            # Download button
            st.download_button(
                "⬇️ Download Predictions CSV",
                submission.to_csv(index=False).encode("utf-8"),
                "LoanPredictions.csv",
                "text/csv"
            )

            # Agreement Analysis (only if all 3 models available)
            if all(k in submission.columns for k in ["LogReg_Pred", "RF_Pred", "XGB_Pred"]):
                submission["All_Agree"] = (submission["LogReg_Pred"] == submission["RF_Pred"]) & \
                                        (submission["RF_Pred"] == submission["XGB_Pred"])
                agreement_rate = submission["All_Agree"].mean() * 100

                st.subheader("🔎 Model Agreement Analysis")
                st.write(f"Agreement between all 3 models: **{agreement_rate:.2f}%**")

                pairwise = {
                    "LogReg vs RF": (submission["LogReg_Pred"] == submission["RF_Pred"]).mean() * 100,
                    "LogReg vs XGB": (submission["LogReg_Pred"] == submission["XGB_Pred"]).mean() * 100,
                    "RF vs XGB": (submission["RF_Pred"] == submission["XGB_Pred"]).mean() * 100,
                }
                st.write("🤝 Pairwise Agreement Rates:", pairwise)

            # Show confidence scores if available
            if "RF_Prob" in submission.columns or "XGB_Prob" in submission.columns:
                st.subheader("🎯 Sample Predictions with Confidence")
                display_cols = ["Loan_ID"]
                if "RF_Pred" in submission.columns: display_cols += ["RF_Pred", "RF_Prob"]
                if "XGB_Pred" in submission.columns: display_cols += ["XGB_Pred", "XGB_Prob"]
                st.dataframe(submission.head(10)[display_cols])

        except Exception as e:
            st.error("⚠️ Please train the models first in the Model Training tab before testing.")
            st.exception(e)


# -------------------------
# 5) Single prediction (placeholder)
# -------------------------
with tabs[4]:
    st.header("🧍 Single Prediction")

    st.markdown("Enter applicant details below to predict Loan Eligibility:")

    # Collect input features (same as training dataset before encoding)
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=150)
    loan_amount_term = st.selectbox("Loan Amount Term (in months)", [360, 120, 180, 240, 300, 84, 60, 480])
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    if st.button("🔮 Predict Eligibility"):
        # Put inputs into a DataFrame
        input_dict = {
            "Gender": [gender],
            "Married": [married],
            "Dependents": [dependents],
            "Education": [education],
            "Self_Employed": [self_employed],
            "ApplicantIncome": [applicant_income],
            "CoapplicantIncome": [coapplicant_income],
            "LoanAmount": [loan_amount],
            "Loan_Amount_Term": [loan_amount_term],
            "Credit_History": [credit_history],
            "Property_Area": [property_area]
        }
        input_df = pd.DataFrame(input_dict)

        # Apply same preprocessing as training
        input_df = pd.get_dummies(input_df)

        # Drop the extra columns we dropped during training
        drop_cols = ['Gender_Female','Married_No','Education_Not Graduate','Self_Employed_No']
        input_df = input_df.drop([c for c in drop_cols if c in input_df.columns], axis=1)

        # Rename columns to match
        rename_cols = {
            'Married_Yes': 'Married',
            'Self_Employed_Yes': 'Self_Employed'
        }
        input_df.rename(columns=rename_cols, inplace=True)

        # Add missing columns
        for c in st.session_state.feature_cols:
            if c not in input_df.columns:
                input_df[c] = 0

        # Ensure correct order
        input_df = input_df[st.session_state.feature_cols]

        # Scale using same scaler
        input_scaled = scaler.transform(input_df)

        # Predictions from models
        if "log_reg" in st.session_state and st.session_state.log_reg is not None:
            log_pred = st.session_state.log_reg.predict(input_scaled)[0]
            log_prob = float(st.session_state.log_reg.predict_proba(input_scaled)[0][1])
            st.write(f"**Logistic Regression:** {'Eligible' if log_pred==1 else 'Not Eligible'}")
            st.write(f"🤖 Logistic Regression Confidence: {log_prob:.2f}")
            st.progress(log_prob)

        # Random Forest
        if "rf" in st.session_state and st.session_state.rf is not None:
            rf_pred = st.session_state.rf.predict(input_scaled)[0]
            rf_prob = float(st.session_state.rf.predict_proba(input_scaled)[0][1])
            st.write(f"**Random Forest:** {'Eligible' if rf_pred==1 else 'Not Eligible'}")
            st.write(f"🌲 Random Forest Confidence: {rf_prob:.2f}")
            st.progress(rf_prob)

        # XGBoost
        if "xgb" in st.session_state and st.session_state.xgb is not None:
            xgb_pred = st.session_state.xgb.predict(input_scaled)[0]
            xgb_prob = float(st.session_state.xgb.predict_proba(input_scaled)[0][1])
            st.write(f"**XGBoost:** {'Eligible' if xgb_pred==1 else 'Not Eligible'}")
            st.write(f"🚀 XGBoost Confidence: {xgb_prob:.2f}")
            st.progress(xgb_prob)

# -------------------------
# 6) Interpretability (placeholder)
# -------------------------
with tabs[5]:
    st.header("🔎 Feature Importance & Interpretability")

    # Helper function to plot feature importance
    def plot_feature_importance(model, feature_names, model_name="Model"):
        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_
            features = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importances
            }).sort_values(by="Importance", ascending=False)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(
                x="Importance", 
                y="Feature", 
                data=features.head(15), 
                palette="viridis", 
                ax=ax
            )
            ax.set_title(f"Top 15 Important Features - {model_name}", fontsize=14)
            st.pyplot(fig)
        else:
            st.warning(f"{model_name} does not provide feature importances.")
    if st.session_state.log_reg is not None:
        st.subheader("🌲 Logistic Regression Feature Importance")
        st.write("Logistic Regression does not provide feature importances directly.")
    # Random Forest Importance
    if st.session_state.rf is not None:
        st.subheader("🌲 Random Forest Feature Importance")
        plot_feature_importance(st.session_state.rf, st.session_state.feature_cols, "Random Forest")

        # XGBoost Importance
    if st.session_state.xgb is not None:
        st.subheader("🚀 XGBoost Feature Importance")
        plot_feature_importance(st.session_state.xgb, st.session_state.feature_cols, "XGBoost")

# -------------------------
# 7) Insights (placeholder)
# -------------------------
with tabs[6]:
    st.header("📊 Final Insights")

    st.markdown("""
    ### 🌲 Random Forest
    - **Applicant Income** and **Loan Amount** are the top deciding factors.  
    - **Coapplicant Income** and **Property Area (Semiurban)** also play significant roles.  
    - **Marital status** has moderate impact.  
    - **Property Area (Rural/Urban)**, **Loan Term**, and **Credit History** contribute but less compared to income & loan size.  
    - **Gender, Dependents, Education** are minimal influencers.  

    ### 🚀 XGBoost
    - **Property Area (Semiurban)** is the most influential factor, followed by **Marital Status**.  
    - **Property Area (Rural & Urban)** also matter, showing strong geographic influence.  
    - **Applicant Income, Coapplicant Income, and Loan Amount** remain important but secondary to property location & marital status.  
    - **Credit History, Loan Term, Gender, Dependents, Education** have relatively small impact.  

    ### 📌 Overall Insights
    - **Location (Property Area)** and **Marital Status** are stronger predictors than expected, especially in XGBoost.  
    - **Income and Loan Amount** are crucial in Random Forest but less dominant in XGBoost.  
    - **Credit History**, while traditionally important, appears less impactful in this dataset — likely due to imbalanced distribution.  
    - **Demographics (Gender, Education, Dependents)** contribute minimally to loan approval decisions.  
    - Among models, **XGBoost** better captures non-linear relationships and interactions, making it the most suitable for deployment.  
    """)