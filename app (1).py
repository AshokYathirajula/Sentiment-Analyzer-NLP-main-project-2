import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="✨ Sentiment Analyzer",
    page_icon="✨",
    layout="centered"
)

# ==========================================
# LOAD MODEL
# ==========================================

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# ==========================================
# TEXT CLEANING
# ==========================================

stop_words = set(stopwords.words("english"))
stop_words = stop_words - {"not", "no", "nor", "never"}

lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ==========================================
# HISTORY
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("✨ Sentiment Analyzer")

    st.markdown("---")

    st.subheader("Model Information")

    st.write("Algorithm : Logistic Regression")
    st.write("Vectorizer : TF-IDF")

    st.markdown("---")

    st.subheader("Examples")

    if st.button("😊 Positive Example"):
        st.session_state.review = (
            "Excellent camera quality and battery backup."
        )

    if st.button("😐 Neutral Example"):
        st.session_state.review = (
            "The product is average."
        )

    if st.button("😞 Negative Example"):
        st.session_state.review = (
            "Worst experience ever. Completely disappointed."
        )

    st.markdown("---")

    st.info(
        """
Built with

• Streamlit
• Scikit-Learn
• Logistic Regression
• TF-IDF
"""
    )

# ==========================================
# HEADER
# ==========================================

st.title("✨ Sentiment Analyzer")

st.caption(
    "Machine Learning Powered Review Intelligence"
)

# ==========================================
# INPUT
# ==========================================

review = st.text_area(
    "Enter Review",
    value=st.session_state.get("review", ""),
    height=180,
    placeholder="Type customer review..."
)

predict = st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
)

# ==========================================
# PREDICTION
# ==========================================

if predict:

    if review.strip() == "":
        st.warning("Please enter some text.")
        st.stop()

    cleaned_review = clean_text(review)

    vector = tfidf.transform([cleaned_review])

    prediction = model.predict(vector)

    probabilities = model.predict_proba(vector)[0]

    sentiment = label_encoder.inverse_transform(
        prediction
    )[0]

    confidence = np.max(probabilities) * 100

    # Emoji

    if sentiment == "Positive":
        emoji = "😊"
        st.success("Positive Sentiment")

    elif sentiment == "Negative":
        emoji = "😞"
        st.error("Negative Sentiment")

    else:
        emoji = "😐"
        st.warning("Neutral Sentiment")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Sentiment",
            f"{emoji} {sentiment}"
        )

    with col2:
        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    # ======================================
    # PROGRESS BAR
    # ======================================

    st.subheader("Confidence Level")

    st.progress(float(confidence / 100))

    # ======================================
    # PROBABILITIES
    # ======================================

    st.subheader("Sentiment Probabilities")

    classes = label_encoder.classes_

    prob_df = pd.DataFrame(
        {
            "Sentiment": classes,
            "Probability (%)":
            probabilities * 100
        }
    )

    st.bar_chart(
        prob_df.set_index("Sentiment")
    )

    # ======================================
    # PIE CHART
    # ======================================

    fig = px.pie(
        prob_df,
        values="Probability (%)",
        names="Sentiment",
        hole=0.5,
        title="Probability Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ======================================
    # TABLE
    # ======================================

    st.subheader("Detailed Analysis")

    st.dataframe(
        prob_df.round(2),
        use_container_width=True
    )

    # ======================================
    # HISTORY
    # ======================================

    st.session_state.history.append(
        {
            "Review": review,
            "Sentiment": sentiment,
            "Confidence":
            round(confidence, 2)
        }
    )

# ==========================================
# HISTORY TABLE
# ==========================================

if len(st.session_state.history) > 0:

    st.subheader("📜 Prediction History")

    st.dataframe(
        pd.DataFrame(
            st.session_state.history
        ),
        use_container_width=True
    )

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Built with ❤️ using Streamlit and Logistic Regression"
)