import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Must be the first Streamlit command!
st.set_page_config(page_title="Emotion Detector", layout="wide")

# 1. Load dataset
def load_dataset(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if ';' in line:
                parts = line.strip().split(';')
                if len(parts) == 2:
                    texts.append(parts[0])
                    labels.append(parts[1])
    return pd.DataFrame({'text': texts, 'label': labels})

# 2. Train model (cache with new decorator)
@st.cache_data(show_spinner=True)
def train_model():
    df = load_dataset("train.txt")
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    return df, vectorizer, model, X_test, y_test, y_pred

# 3. Main App
st.title("Emotion Detection from Text")

df, vectorizer, model, X_test, y_test, y_pred = train_model()

# Sidebar Input
user_input = st.text_input("Enter a sentence to analyze emotion:")

if user_input:
    vec = vectorizer.transform([user_input])
    pred = model.predict(vec)[0]
    pred_proba = model.predict_proba(vec)[0]
    prob = pred_proba.max()
    st.subheader("Prediction Result")
    st.write(f"**Predicted Emotion:** {pred}")
    st.write(f"**Confidence:** {prob:.2f}")

    # Bar plot of prediction probabilities
    classes = model.classes_
    fig1, ax1 = plt.subplots(figsize=(8,4))
    sns.barplot(x=classes, y=pred_proba, ax=ax1, palette='viridis')
    ax1.set_ylabel("Probability")
    ax1.set_xlabel("Emotion Class")
    ax1.set_title("Prediction Confidence per Emotion Class")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

# Data Visualization
st.subheader("Emotion Distribution in Dataset")
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.countplot(data=df, x='label', order=df['label'].value_counts().index, palette='viridis', ax=ax2)
plt.xticks(rotation=45)
st.pyplot(fig2)


# Confusion Matrix
st.subheader("Confusion Matrix")
labels_sorted = sorted(model.classes_)
cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels_sorted, yticklabels=labels_sorted, cmap='Blues', ax=ax4)
ax4.set_xlabel("Predicted")
ax4.set_ylabel("Actual")
st.pyplot(fig4)

# Metrics
st.subheader("Evaluation Metrics")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
st.write(f"**Accuracy:**  {accuracy:.4f}")
st.write(f"**Precision:** {precision:.4f}")
st.write(f"**Recall:**    {recall:.4f}")
st.write(f"**F1 Score:**  {f1:.4f}")
