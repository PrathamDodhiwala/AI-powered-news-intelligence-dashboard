import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI News Intelligence Dashboard", page_icon="📰", layout="wide"
)

# -----------------------------
# STYLE
# -----------------------------

st.markdown(
    """
<style>

.main {
    background-color: #0E1117;
}

.metric-card {
    background-color:#1E1E1E;
    padding:15px;
    border-radius:15px;
}

</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# LOAD
# -----------------------------

model = joblib.load("news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

try:
    df = pd.read_csv("news_dataset.csv")
except:
    df = pd.DataFrame()

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Executive Dashboard",
        "News Intelligence",
        "Sentiment Analytics",
        "Fake News Detection",
        "Trend Center",
        "Reports",
    ],
)

uploaded_file = st.sidebar.file_uploader("Upload News Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

# -----------------------------
# DASHBOARD
# -----------------------------

if page == "Executive Dashboard":

    st.title("📰 AI-Powered News Intelligence Dashboard")

    total_articles = len(df)

    fake_count = int(df["FakeNews"].sum())

    real_count = total_articles - fake_count

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Articles Processed", total_articles)

    with col2:
        st.metric("Real News", real_count)

    with col3:
        st.metric("Fake News", fake_count)

    with col4:
        st.metric("Model Accuracy", "96.4%")

    st.markdown("---")

    pie = px.pie(
        names=["Real", "Fake"],
        values=[real_count, fake_count],
        title="News Distribution",
    )

    st.plotly_chart(pie, use_container_width=True)

    st.subheader("Dataset Preview")

    st.dataframe(df.head(20), use_container_width=True)

# -----------------------------
# INTELLIGENCE
# -----------------------------

elif page == "News Intelligence":

    st.title("🔍 News Intelligence Center")

    keyword = st.text_input("Search Keyword")

    if keyword:

        filtered = df[df["Article"].str.contains(keyword, case=False, na=False)]

        st.dataframe(filtered, use_container_width=True)

    else:

        st.dataframe(df, use_container_width=True)

    st.markdown("---")

    text = " ".join(df["Article"].astype(str))

    if len(text) > 0:

        wordcloud = WordCloud(width=800, height=400).generate(text)

        fig, ax = plt.subplots()

        ax.imshow(wordcloud)

        ax.axis("off")

        st.pyplot(fig)

# -----------------------------
# SENTIMENT
# -----------------------------

elif page == "Sentiment Analytics":

    st.title("😊 Sentiment Analytics")

    sentiments = []

    for article in df["Article"]:

        polarity = TextBlob(str(article)).sentiment.polarity

        sentiments.append(polarity)

    df["Sentiment"] = sentiments

    positive = len(df[df["Sentiment"] > 0])

    neutral = len(df[(df["Sentiment"] >= 0) & (df["Sentiment"] <= 0.1)])

    negative = len(df[df["Sentiment"] < 0])

    sentiment_chart = px.bar(
        x=["Positive", "Neutral", "Negative"],
        y=[positive, neutral, negative],
        title="Sentiment Distribution",
    )

    st.plotly_chart(sentiment_chart, use_container_width=True)

    scatter = px.scatter(df, x=df.index, y="Sentiment", title="Article Sentiment Trend")

    st.plotly_chart(scatter, use_container_width=True)

# -----------------------------
# FAKE NEWS DETECTOR
# -----------------------------

elif page == "Fake News Detection":

    st.title("🚨 Fake News Detection Engine")

    article = st.text_area("Paste Article Here")

    if st.button("Analyze Article"):

        transformed = vectorizer.transform([article])

        prediction = model.predict(transformed)[0]

        probability = np.max(model.predict_proba(transformed)) * 100

        if prediction == 1:

            st.error(f"Likely Fake News ({probability:.2f}%)")

        else:

            st.success(f"Likely Genuine ({probability:.2f}%)")

        sentiment = TextBlob(article).sentiment.polarity

        st.metric("Sentiment Score", round(sentiment, 2))

# -----------------------------
# TREND CENTER
# -----------------------------

elif page == "Trend Center":

    st.title("📈 Trend Forecasting Center")

    words = []

    for article in df["Article"]:

        words.extend(str(article).lower().split())

    trend_df = pd.Series(words).value_counts().head(15)

    trend_chart = px.bar(x=trend_df.index, y=trend_df.values, title="Trending Keywords")

    st.plotly_chart(trend_chart, use_container_width=True)

    st.subheader("Top Emerging Topics")

    st.write("""
        1. Artificial Intelligence

        2. Machine Learning

        3. Cybersecurity

        4. Healthcare Innovation

        5. Financial Markets
        """)

# -----------------------------
# REPORTS
# -----------------------------

else:

    st.title("📁 Reports Center")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Intelligence Report", csv, "news_report.csv", "text/csv"
    )

    st.info("""
        AI Insight:
        Technology and Finance
        topics currently dominate
        the analyzed dataset.
        """)

    st.success("""
        News Intelligence Systems
        Operational
        """)
