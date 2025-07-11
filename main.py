import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")

# ===========================================
# ✅ Load & clean data once
# ===========================================
@st.cache_data
def load_data():
    df = pd.read_csv(
        r"C:\Users\ASUS\.cache\kagglehub\datasets\jpmiller\employee-attrition-for-healthcare\versions\3/watson_healthcare_modified.csv"
    )
    df = df.drop(columns=["EmployeeID", "EmployeeCount", "Over18", "StandardHours"])
    df = df.drop_duplicates()
    df["Attrition"] = df["Attrition"].map({"No": 0, "Yes": 1})
    return df

raw_df = load_data()

# ===============================
# ✅ Clean & engineer version
# ===============================
df = raw_df.copy()
df["LogMonthlyIncome"] = np.log1p(df["MonthlyIncome"])
df["YearsInOtherRoles"] = df["YearsAtCompany"] - df["YearsInCurrentRole"]
df["YearsInOtherRoles"] = df["YearsInOtherRoles"].clip(lower=0)
df["LoyaltyRatio"] = df["YearsAtCompany"] / df["TotalWorkingYears"]
df["LoyaltyRatio"] = df["LoyaltyRatio"].replace([np.inf, -np.inf], 0).fillna(0)
df["IncomePerYear"] = (df["MonthlyIncome"] * 12) / df["TotalWorkingYears"]
df["IncomePerYear"] = df["IncomePerYear"].replace([np.inf, -np.inf], 0).fillna(0)

numeric_cols = [
    "Age", "DailyRate", "DistanceFromHome", "LogMonthlyIncome",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsInOtherRoles", "LoyaltyRatio", "IncomePerYear"
]
ordinal_cols = [
    "Education", "EnvironmentSatisfaction", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "PerformanceRating",
    "RelationshipSatisfaction", "TrainingTimesLastYear", "WorkLifeBalance", "Shift"
]
categorical_cols = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime"
]

X = df.drop(columns=["Attrition"])
y = df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

numeric_pipe = Pipeline([("scaler", StandardScaler())])
ordinal_pipe = Pipeline([("ordinal", OrdinalEncoder())])
categorical_pipe = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("ord", ordinal_pipe, ordinal_cols),
    ("cat", categorical_pipe, categorical_cols)
])

# =======================================
# ✅ MODELS: One tuned for score, one for recall
# =======================================
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
xgb = XGBClassifier(verbosity=0, random_state=42)
meta = LogisticRegression()

stacked = StackingClassifier(
    estimators=[('rf', rf), ('xgb', xgb)],
    final_estimator=meta,
    cv=5
)

pipe_A = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", stacked)
])

pipe_A.fit(X_train, y_train)
y_probs_A = pipe_A.predict_proba(X_test)[:, 1]

# high recall version
pipe_B = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", stacked)
])
pipe_B.fit(X_train, y_train)
y_probs_B = pipe_B.predict_proba(X_test)[:, 1]

# auto-threshold for high recall
precision, recall, thresholds = precision_recall_curve(y_test, y_probs_B)
candidates = [(p, r, t) for p, r, t in zip(precision, recall, thresholds) if r >= 0.80]
best_threshold_B = candidates[-1][2] if candidates else 0.5

# ===========================================
# ✅ Streamlit pages
# ===========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to page:", [
    "1️⃣ Raw Data Preview",
    "2️⃣ Raw Data Visuals",
    "3️⃣ Clean Data Preview",
    "4️⃣ Clean Data Visuals",
    "5️⃣ Models & Code",
    "6️⃣ Predict with Model A",
    "7️⃣ Predict with Model B"
])

# ------------------------------
if page == "1️⃣ Raw Data Preview":
    st.title("🔍 Raw Data Preview")
    st.dataframe(raw_df.head(50))

# ------------------------------
elif page == "2️⃣ Raw Data Visuals":
    st.title("📊 Raw Data Visualizations")
    st.write("**Class Balance:**")
    fig, ax = plt.subplots()
    sns.countplot(x=raw_df["Attrition"], ax=ax)
    st.pyplot(fig)

    st.write("**Correlation Heatmap:**")
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(raw_df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig)

# ------------------------------
elif page == "3️⃣ Clean Data Preview":
    st.title("🧹 Cleaned & Engineered Data Preview")
    st.write("Showing a sample of the cleaned dataset:")
    st.dataframe(df.head(50))

# ------------------------------
elif page == "4️⃣ Clean Data Visuals":
    st.title("📈 Cleaned Data Visualizations")
    st.write("**Attrition Balance:**")
    fig, ax = plt.subplots()
    sns.countplot(x=df["Attrition"], ax=ax)
    st.pyplot(fig)

    st.write("**Correlation Heatmap:**")
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig)

# ------------------------------
elif page == "5️⃣ Models & Code":
    st.title("🧮 Model Details & Strategy")
    st.subheader("✅ Model A (Highest Score)")
    st.code("""
    - Model: Stacking (RF + XGB + Logistic)
    - Goal: Best balanced F1/Accuracy
    - Uses SMOTE
    """)

    st.subheader("✅ Model B (Higher True Positive)")
    st.code(f"""
    - Same base model, but:
    - Custom threshold: {best_threshold_B:.2f}
    - Tuned for higher recall ~0.80 to detect more likely leavers
    """)

# ------------------------------
elif page == "6️⃣ Predict with Model A":
    st.title("🤖 Predict with Model A")
    input_data = {}
    for col in numeric_cols:
        input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()), key=f"A_{col}")
    for col in ordinal_cols:
        input_data[col] = st.selectbox(f"{col}", sorted(df[col].unique()), key=f"A_{col}")
    for col in categorical_cols:
        input_data[col] = st.selectbox(f"{col}", sorted(df[col].unique()), key=f"A_{col}")

    input_df = pd.DataFrame([input_data])
    if st.button("Predict with Model A"):
        prob = pipe_A.predict_proba(input_df)[0][1]
        pred = int(prob >= 0.5)
        st.write(f"Probability: {prob:.2f}")
        st.success("Likely to STAY" if pred == 0 else "⚠️ Likely to LEAVE!")

# ------------------------------
elif page == "7️⃣ Predict with Model B":
    st.title("🚨 Predict with Model B (High Recall)")
    input_data = {}
    for col in numeric_cols:
        input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()), key=f"B_{col}")
    for col in ordinal_cols:
        input_data[col] = st.selectbox(f"{col}", sorted(df[col].unique()), key=f"B_{col}")
    for col in categorical_cols:
        input_data[col] = st.selectbox(f"{col}", sorted(df[col].unique()), key=f"B_{col}")

    input_df = pd.DataFrame([input_data])
    if st.button("Predict with Model B"):
        prob = pipe_B.predict_proba(input_df)[0][1]
        pred = int(prob >= best_threshold_B)
        st.write(f"Probability: {prob:.2f}")
        st.success("Likely to STAY" if pred == 0 else "⚠️ Likely to LEAVE!")
