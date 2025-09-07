import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import pandas as pd
import re
from PyPDF2 import PdfReader
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# -------------------------
# Caching Models & Functions
# -------------------------
@st.cache_resource
def load_summarizer():
    model_path = "./t5_model"  # local model folder
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return pipeline("summarization", model=model, tokenizer=tokenizer)

@st.cache_data
def load_pickle():
    return pd.read_pickle("sentiment_model.pkl")

@st.cache_data
def read_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# -------------------------
# Emoji & Cleaning
# -------------------------
emoji_map = {
    "👍": "good", "🔥": "amazing", "❤️": "love", "😂": "funny",
    "👏": "applause", "😊": "happy", "🤔": "uncertain", "💡": "idea",
    "🙏": "request"
}

def emoji_to_text(text):
    for e, w in emoji_map.items():
        text = text.replace(e, " " + w + " ")
    return text

def clean_text(text):
    return re.sub(r'[^\w\s.,!?]', '', text).strip()

# -------------------------
# Sentiment Prediction
# -------------------------
def predict_sentiment(comment, df):
    # Check direct matches from dataset
    match = df[df["Comment"].str.lower() == comment.lower()]
    if not match.empty:
        return match["Sentiment"].values[0]

    # Keywords (weighted: strong ones first)
    pos_keywords = [
        "excellent", "fantastic", "amazing", "wonderful", "brilliant", "superb",
        "love", "good", "happy", "great", "satisfied", "pleased", "support",
        "like", "positive", "improved", "strong"
    ]
    neg_keywords = [
        "terrible", "horrible", "awful", "worst", "disaster", "pathetic",
        "hate", "bad", "poor", "disappointed", "angry", "frustrated",
        "upset", "problem", "issue", "dissatisfied", "negative", "unfair"
    ]
    neutral_keywords = [
        "okay", "fine", "average", "normal", "fair", "moderate", "balanced",
        "reasonable", "satisfactory", "adequate", "typical"
    ]

    comment_lower = comment.lower()

    # Handle negations like "not good" → negative
    for word in pos_keywords:
        if f"not {word}" in comment_lower:
            return -1
    for word in neg_keywords:
        if f"not {word}" in comment_lower:
            return 1

    # Initialize scores
    pos_score, neg_score, neu_score = 0, 0, 0

    # Count positive/negative/neutral signals
    for i, word in enumerate(pos_keywords):
        if word in comment_lower:
            pos_score += 2 if i < 5 else 1  # strong words weighted higher
    for i, word in enumerate(neg_keywords):
        if word in comment_lower:
            neg_score += 2 if i < 5 else 1
    for word in neutral_keywords:
        if word in comment_lower:
            neu_score += 1

    # Debugging: see how many words contributed
    # print(comment, pos_score, neg_score, neu_score)

    # Decision based on scores
    if pos_score > neg_score and pos_score > neu_score:
        return 1  # Positive
    elif neg_score > pos_score and neg_score > neu_score:
        return -1  # Negative
    elif neu_score > max(pos_score, neg_score):
        return 0  # Neutral
    else:
        # If tie → pick stronger side
        if pos_score > neg_score:
            return 1
        elif neg_score > pos_score:
            return -1
        else:
            return 0  # fallback neutral

# -------------------------
# Word Cloud Generator
# -------------------------
def generate_wordcloud(comments, sentiment_name, color="white"):
    text = " ".join(comments)
    if not text:
        return None
    wc = WordCloud(width=800, height=400, background_color=color, colormap="viridis").generate(text)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"{sentiment_name} Word Cloud", fontsize=14)
    return fig

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Government Consultation Analyzer", layout="wide")
st.title("📑 Government Consultation Comment Analyzer")

uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])
comments = []

sentiment_df = load_pickle()  # cached pickle

if uploaded_pdf:
    pdf_text = read_pdf(uploaded_pdf)
    comments = [clean_text(emoji_to_text(c)) for c in pdf_text.split("\n") if c.strip()]

if st.button("Run Analysis"):
    if not comments:
        st.warning("No comments found in PDF.")
    else:
        sentiments = [predict_sentiment(c, sentiment_df) for c in comments]
        df = pd.DataFrame({"Comment": comments, "Sentiment": sentiments})
        sentiment_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
        df["Sentiment_Label"] = df["Sentiment"].map(sentiment_map)

        st.subheader("📊 Analysis Dashboard")
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Data", "📈 Charts", "☁ Word Clouds", "📝 Summary"])

        with tab1:
            st.dataframe(df[["Comment", "Sentiment_Label"]], use_container_width=True, height=400)
            st.download_button("📥 Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                               file_name="sentiment_results.csv", mime="text/csv")

        with tab2:
            sentiment_counts = df["Sentiment_Label"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]

            fig_pie = px.pie(sentiment_counts, names="Sentiment", values="Count",
                             color="Sentiment",
                             color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"},
                             hole=0.4, title="Sentiment Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

            fig_bar = px.bar(sentiment_counts, x="Sentiment", y="Count", color="Sentiment",
                             color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"},
                             text="Count", title="Sentiment Count Comparison")
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

        with tab3:
            for s, label, color in [(1, "Positive", "white"), (0, "Neutral", "black"), (-1, "Negative", "white")]:
                comments_subset = df[df["Sentiment"] == s]["Comment"].tolist()
                wc_fig = generate_wordcloud(comments_subset, label, color)
                if wc_fig:
                    st.pyplot(wc_fig)

        with tab4:
            summarizer = load_summarizer()  # cached
            draft_summary = " ".join(comments[:20])  # take first 20 comments for context
            input_text = "summarize: " + draft_summary
            final_summary = summarizer(input_text, max_length=200, min_length=60, do_sample=False)[0]["summary_text"]

            st.markdown("### 📝 Overall Summary")
            st.write(final_summary)
