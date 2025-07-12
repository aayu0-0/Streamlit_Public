import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Predict", "📊 Visualize"])

# ----------------- Home -----------------
if page == "🏠 Home":
    st.title("🌼 Iris Flower Classification")
    st.write("""
    Welcome to the Iris Flower Classification App!

    **Features:**
    - Predict the species of an Iris flower by entering measurements.
    - Explore the Iris dataset visually.

    Use the sidebar to navigate.
    """)

# ----------------- Predict -----------------
elif page == "🔍 Predict":
    st.title("🔍 Predict Iris Flower Species")

    sepal_length = st.number_input("Sepal length (cm)", 0.0, 10.0, 5.1)
    sepal_width = st.number_input("Sepal width (cm)", 0.0, 10.0, 3.5)
    petal_length = st.number_input("Petal length (cm)", 0.0, 10.0, 1.4)
    petal_width = st.number_input("Petal width (cm)", 0.0, 10.0, 0.2)

    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=iris.feature_names)

    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.write(f"## 🌸 Prediction: **{iris.target_names[prediction][0].capitalize()}**")

    st.write("### 🔬 Prediction Probabilities:")
    proba_df = pd.DataFrame({
        'Species': iris.target_names,
        'Probability': prediction_proba[0]
    })

    st.dataframe(proba_df)

    # Optional pie chart
    fig, ax = plt.subplots()
    ax.pie(prediction_proba[0], labels=iris.target_names, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

# ----------------- Visualize -----------------
elif page == "📊 Visualize":
    st.title("📊 Iris Dataset Visualization")

    df = X.copy()
    df['species'] = pd.Categorical.from_codes(y, iris.target_names)

    st.write("### First 5 rows of the dataset:")
    st.dataframe(df.head())

    st.write("### Pairplot of features colored by species:")
    fig = sns.pairplot(df, hue="species")
    st.pyplot(fig)
