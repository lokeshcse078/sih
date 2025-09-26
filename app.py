import pandas as pd
import re
from PyPDF2 import PdfReader
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt  # Only for WordCloud
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect
import langcodes
from deep_translator import GoogleTranslator
import base64
import requests
from io import BytesIO
import bcrypt
from datetime import datetime
import random
import smtplib
from email.message import EmailMessage
from supabase import create_client

# ------------------------- Supabase Setup -------------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------- OTP Email -------------------------
OTP_STORE = {}  # Temporary in-memory store for OTPs
SMTP_EMAIL = st.secrets["SMTP_EMAIL"]
SMTP_PASSWORD = st.secrets["SMTP_PASSWORD"]

def send_otp(email):
    otp = random.randint(100000, 999999)
    st.session_state.otp_store = st.session_state.get("otp_store", {})
    st.session_state.otp_store[email] = otp
    
    msg = EmailMessage()
    msg['Subject'] = "Your OTP for Registration"
    msg['From'] = SMTP_EMAIL
    msg['To'] = email
    msg.set_content(f"Your OTP for registration is: {otp}")
    
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(SMTP_EMAIL, SMTP_PASSWORD)
        smtp.send_message(msg)

    return otp

def verify_otp(email, entered_otp):
    otp_store = st.session_state.get("otp_store", {})
    return otp_store.get(email) == entered_otp


# ------------------------- Model Loading -------------------------
@st.cache_resource
def load_summarizer():
    model_id = "https://huggingface.co/lk10308/t5"  
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer

@st.cache_data
def load_pickle():
    url = "https://github.com/lokeshcse078/sih/blob/main/sentiment_model.pkl"
    r = requests.get(url)
    return pd.read_pickle(BytesIO(r.content))

# ------------------------- PDF / TXT Reading -------------------------
def read_pdf(file):
    pdf = PdfReader(file)
    text = "".join([page.extract_text() + "\n" for page in pdf.pages])
    return text

def read_txt(file):
    return file.read().decode("utf-8")

def read_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".txt"):
        return read_txt(uploaded_file)
    else:
        st.markdown('<div class="black-warning">Unsupported file type.</div>')
        return ""

# ------------------------- Emoji & Cleaning -------------------------
emoji_map = {
    "👍":"good","👌":"perfect","😊":"happy","😄":"smile","😁":"grin","🤩":"excited","😎":"cool",
    "🎉":"celebration","🥳":"party","🙌":"cheer","👏":"applause","❤️":"love","💖":"love","💗":"love",
    "💪":"strong","💃":"dance","🕺":"dance","🌟":"star","🔥":"amazing","💯":"perfect",
    "🤔":"uncertain","😐":"neutral","😶":"silent","😑":"expressionless","🤷":"confused",
    "😬":"awkward","😅":"nervous","🫣":"peek","😇":"innocent",
    "👎":"bad","😢":"sad","😭":"cry","😡":"angry","🤬":"angry","💔":"heartbreak","😞":"disappointed",
    "😔":"sad","😖":"frustrated","😫":"tired","😩":"frustrated","😤":"angry","💀":"dead","☠️":"dead",
    "😱":"shock","😨":"fear","😰":"worry","😓":"stress",
    "😂":"funny","🤣":"funny","😜":"playful","😝":"playful","🤪":"crazy","🤯":"mindblown","😲":"surprised",
    "😳":"embarrassed","🙃":"funny",
    "🤝":"agreement","✌️":"victory","🤟":"love","🤞":"hope","🙏":"request","💌":"love","💬":"message",
    "🫂":"hug","🤲":"offer","🖐️":"stop",
    "🌈":"rainbow","🌞":"sun","☀️":"sun","🌙":"moon","⭐":"star","💡":"idea","📢":"announcement",
    "🎁":"gift","🎶":"music","🎵":"music"
}

def emoji_to_text(text):
    for e, w in emoji_map.items():
        text = text.replace(e, " " + w + " ")
    return text

def clean_text(text):
    return re.sub(r'[^\w\s.,!?]', '', text).strip()

# ------------------------- Sentiment Prediction -------------------------
def predict_sentiment(comment, df):
    comment_lower = comment.lower().strip()
    sentences = re.split(r'[.!?]\s*', comment_lower)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    pos_score, neg_score, neu_score = 0,0,0
    
    for sentence in sentences:
        match = df[df["Comment"].str.lower() == sentence]
        if not match.empty:
            sentiment_value = match["Sentiment"].values[0]
            if sentiment_value == 1:
                pos_score += 2
            elif sentiment_value == -1:
                neg_score += 2
            else:
                neu_score += 1
    
    pos_keywords = ["excellent","fantastic","amazing","wonderful","brilliant","superb",
                    "love","good","happy","great","satisfied","pleased","support","like",
                    "positive","improved","strong","helpful","efficient","outstanding","best","smooth"]
    neg_keywords = ["terrible","horrible","awful","worst","disaster","pathetic","hate","bad","poor",
                    "disappointed","angry","frustrated","upset","problem","issue","dissatisfied","negative",
                    "unfair","slow","delay","buggy","problematic","error","crash"]
    neutral_keywords = ["okay","fine","average","normal","fair","moderate","balanced",
                        "reasonable","satisfactory","adequate","typical","standard"]
    
    for word in pos_keywords:
        if f"not {word}" in comment_lower: neg_score += 2
    for word in neg_keywords:
        if f"not {word}" in comment_lower: pos_score += 2
    
    for i, word in enumerate(pos_keywords):
        if word in comment_lower: pos_score += 2 if i < 5 else 1
    for i, word in enumerate(neg_keywords):
        if word in comment_lower: neg_score += 2 if i < 5 else 1
    for word in neutral_keywords:
        if word in comment_lower: neu_score += 1
    
    if pos_score > neg_score and pos_score > neu_score:
        return 1
    elif neg_score > pos_score and neg_score > neu_score:
        return -1
    elif neu_score > max(pos_score, neg_score):
        return 0
    else:
        return 1 if pos_score > neg_score else (-1 if neg_score > pos_score else 0)

# ------------------------- Word Cloud -------------------------
def generate_wordcloud(comments, sentiment_name, bg_color="white"):
    text = " ".join(comments)
    if not text: return None
    
    colormap = "Dark2" if bg_color=="white" else "viridis"
    contour_color = "black" if bg_color=="white" else None
    contour_width = 1 if bg_color=="white" else 0
    
    wc = WordCloud(width=800, height=400, background_color=bg_color,
                   colormap=colormap, contour_color=contour_color, contour_width=contour_width).generate(text)
    
    fig, ax = plt.subplots(figsize=(8,4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"{sentiment_name} Word Cloud", fontsize=16, fontweight="bold", pad=12)
    return fig

# ------------------------- Translation -------------------------
def translate_to_english(text):
    try:
        lang_code = detect(text)
        lang_full = langcodes.get(lang_code).language_name().capitalize() if lang_code else "Unknown"
        if lang_code != "en":
            translated_text = GoogleTranslator(source="auto", target="en").translate(text)
            return translated_text, lang_full
        return text, "English"
    except:
        return text, "Unknown"

# ------------------------- Streamlit UI -------------------------
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_base64("download.jpeg")

st.markdown("""
<style>
div.stButton > button:first-child {background-color: #002147; color: white !important; font-weight: bold; border-radius: 8px; padding: 10px 20px;}
div.stButton > button:hover {background-color: #004080; color: #ffffff !important;}
div.stdownload_button > button {background-color: #002147; color: white !important; font-weight: bold; border-radius: 8px; padding: 10px 20px; margin-top: 10px;}
div.stdownload_button > button:hover {background-color: #004080; color: #ffffff !important;}
</style>
""", unsafe_allow_html=True)

page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{background-color: #D3D3D3; color: #000; font-family: "Segoe UI", "Arial", sans-serif;}}
[data-testid="stSidebar"] {{background-color: #002147; color: white;}}
[data-testid="stSidebar"] * {{color: white !important;}}
.header-bar {{display: flex; align-items: center; justify-content: flex-start; background: linear-gradient(to right, #FF9933, #FFFFFF, #138808); padding: 12px 20px; border-bottom: 4px solid #002147;}}
.header-bar img {{height: 55px; margin-right: 18px;}}
.header-bar h1 {{font-size: 26px; font-weight: bold; font-family: "Georgia", "Times New Roman", serif; color: #002147; margin:0; letter-spacing:0.6px;}}
.black-warning {{background-color: #000; color: #fff; padding: 12px 18px; border-radius:8px; font-size:16px; font-weight:500; margin:15px 0; border-left:6px solid #FFCC00;}}
</style>

<div class="header-bar">
    <img src="data:image/png;base64,{logo_base64}">
    <h1>Government Sentiment Analysis Dashboard</h1>
</div>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ================= Login/Register with OTP =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "otp_verified" not in st.session_state:
    st.session_state.otp_verified = False

if not st.session_state.logged_in:
    tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

    with tab_login:
        user_email = st.text_input("Email")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            try:
                res = supabase.auth.sign_in_with_password({"email": user_email, "password": pwd})
                if res.user is not None:
                    st.session_state.logged_in = True
                    st.session_state.username = user_email
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid email or password")
            except:
                st.error("Login failed. Please try again.")

    with tab_register:
        new_email = st.text_input("New Email")
        new_pwd = st.text_input("New Password", type="password")
        if st.button("Send OTP"):
            try:
                send_otp(new_email)
                st.session_state.temp_email = new_email
                st.session_state.temp_pwd = new_pwd
                st.success(f"OTP sent to {new_email}")
            except:
                st.error("Failed to send OTP. Check email settings.")

        if "temp_email" in st.session_state:
            otp_input = st.text_input("Enter OTP")
            if st.button("Verify OTP"):
                if verify_otp(st.session_state.temp_email, int(otp_input)):
                    try:
                        # Step 1: Register in Supabase Auth
                        res = supabase.auth.sign_up({
                            "email": st.session_state.temp_email,
                            "password": st.session_state.temp_pwd
                        })
                
                        if res.user is not None:
                            # Step 2: Hash the password before storing in DB
                            hashed_pwd = bcrypt.hashpw(st.session_state.temp_pwd.encode(), bcrypt.gensalt()).decode()
                
                            # Step 3: Insert into custom 'users' table
                            supabase.table("sih").insert({
                                "email": st.session_state.temp_email,
                                "password_hash": hashed_pwd,
                                "created_at": datetime.utcnow().isoformat()
                            }).execute()
                
                            st.success("Registration successful! Please login.")
                            # Cleanup
                            del st.session_state.temp_email
                            del st.session_state.temp_pwd
                        else:
                            st.error("Registration failed. Email may already exist.")
                    except Exception as e:
                        st.error(f"Error registering user in Supabase: {e}")
                else:
                    st.error("Invalid OTP")

else:
    st.sidebar.success(f"Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
    uploaded_file = st.file_uploader("Upload PDF or TXT file", type=["pdf","txt"])
    comments=[]
    sentiment_df = load_pickle()

    if uploaded_file:
        file_text = read_file(uploaded_file)
        raw_comments = [c.strip() for c in file_text.split("\n") if c.strip()]
        processed_comments, detected_langs, translated_comments = [], [], []

        for c in raw_comments:
            c = emoji_to_text(c)
            clean_c = clean_text(c)
            translated, lang = translate_to_english(clean_c)
            processed_comments.append(clean_c)
            translated_comments.append(translated)
            detected_langs.append(lang)

        comments = processed_comments

        if st.button("Run Analysis"):
            if not comments:
                st.markdown('<div class="black-warning">No comments found in file.</div>')
            else:
                sentiments = [predict_sentiment(c,sentiment_df) for c in translated_comments]
                df = pd.DataFrame({
                    "Original_Comment": comments,
                    "Language": detected_langs,
                    "Translated_Comment": translated_comments,
                    "Sentiment": sentiments
                })
                sentiment_map={1:"Positive",0:"Neutral",-1:"Negative"}
                df["Sentiment_Label"] = df["Sentiment"].map(sentiment_map)

                st.subheader("📊 Analysis Dashboard")
                tab1, tab2, tab3, tab4 = st.tabs(["📋 Data","📈 Charts","☁ Word Clouds","📝 Summary"])

                with tab1:
                    st.dataframe(df[["Original_Comment","Language","Sentiment_Label"]], use_container_width=True, height=400)
                    st.download_button("📥 Download CSV", data=df.to_csv(index=False).encode("utf-8"),
                                       file_name="sentiment_results.csv", mime="text/csv")

                with tab2:
                    # Sentiment Pie & Bar
                    sentiment_counts = df["Sentiment_Label"].value_counts().reset_index()
                    sentiment_counts.columns = ["Sentiment", "Count"]

                    fig_pie = alt.Chart(sentiment_counts).mark_arc(innerRadius=50).encode(
                        theta="Count:Q",
                        color=alt.Color("Sentiment:N", scale=alt.Scale(domain=["Positive","Neutral","Negative"],
                                                                        range=["green","gray","red"])),
                        tooltip=["Sentiment","Count"]
                    )
                    st.altair_chart(fig_pie, use_container_width=True)

                    fig_bar = alt.Chart(sentiment_counts).mark_bar().encode(
                        x="Sentiment:N",
                        y="Count:Q",
                        color=alt.Color("Sentiment:N", scale=alt.Scale(domain=["Positive","Neutral","Negative"],
                                                                        range=["green","gray","red"])),
                        tooltip=["Sentiment","Count"]
                    )
                    st.altair_chart(fig_bar, use_container_width=True)

                    # Language Distribution
                    lang_counts = df["Language"].value_counts().reset_index()
                    lang_counts.columns = ["Language", "Count"]

                    fig_lang_bar = alt.Chart(lang_counts).mark_bar().encode(
                        x="Language:N",
                        y="Count:Q",
                        color="Language:N",
                        tooltip=["Language","Count"]
                    )
                    st.altair_chart(fig_lang_bar, use_container_width=True)

                    # Combined Sentiment + Language
                    combined_counts = df.groupby(["Language","Sentiment_Label"]).size().reset_index(name="Count")
                    fig_combined = alt.Chart(combined_counts).mark_bar().encode(
                        x="Language:N",
                        y="Count:Q",
                        color=alt.Color("Sentiment_Label:N", scale=alt.Scale(domain=["Positive","Neutral","Negative"],
                                                                             range=["green","gray","red"])),
                        tooltip=["Language","Sentiment_Label","Count"]
                    )
                    st.altair_chart(fig_combined, use_container_width=True)

                with tab3:
                    for sentiment in ["Positive","Neutral","Negative"]:
                        subset = df[df["Sentiment_Label"]==sentiment]["Translated_Comment"].tolist()
                        if subset:
                            fig_wc = generate_wordcloud(subset, sentiment)
                            st.pyplot(fig_wc)
                with tab4:
                    summarizer = load_summarizer()
                    draft_summary = " ".join(df["Translated_Comment"].tolist()[:20])
                    input_text = "summarize: " + draft_summary
                    final_summary = summarizer(input_text, max_length=200, min_length=60, do_sample=False)[0]["summary_text"]

                    st.markdown('<div class="black-warning">### 📝 Overall Summary (based on translated comments)</div>',unsafe_allow_html=True)
                    st.write(final_summary)
        else:
            st.markdown(
            '<div class="black-warning">⚠ Please upload a PDF or TXT file to proceed.</div>',
            unsafe_allow_html=True)
                















