import streamlit as st
import whisper
from transformers import pipeline
import torch
import matplotlib.pyplot as plt
import os

# Set Streamlit page config
st.set_page_config(page_title="VoxSumm", layout="wide")

# Display Title
st.title("🎙️ VoxSumm - Hindi Speech Summarizer & Sentiment Analyzer")
st.write("Upload a Hindi audio file to transcribe, translate, summarize, and analyze its sentiment.")

# Load models efficiently
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model("medium")
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en", device=0 if device == "cuda" else -1)
    summarizer = pipeline("summarization", model="facebook/mbart-large-50", device=0 if device == "cuda" else -1)
    sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=0 if device == "cuda" else -1)
    return whisper_model, translator, summarizer, sentiment_analyzer

# Load models once
whisper_model, translator, summarizer, sentiment_analyzer = load_models()

# File upload
uploaded_file = st.file_uploader("Upload a Hindi audio file", type=["wav", "mp3", "m4a"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    # Save file temporarily
    file_path = "temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Step 1: Transcription
    st.write("📜 **Step 1: Transcribing Audio...**")
    transcription = whisper_model.transcribe(file_path, language="hi")["text"]
    st.success("✅ Transcription Completed!")
    st.text_area("📌 **Transcribed Hindi Text:**", transcription, height=150)

    # Step 2: Translation
    st.write("🌍 **Step 2: Translating to English...**")
    try:
        translation = translator(transcription, max_length=512)[0]['translation_text']
        st.success("✅ Translation Completed!")
        st.text_area("📌 **Translated English Text:**", translation, height=150)
    except Exception as e:
        st.error(f"❌ Translation Failed: {e}")
        translation = None

    # Step 3: Summarization
    if translation:
        st.write("✍️ **Step 3: Summarizing the Text...**")
        try:
            summary = summarizer(translation, max_length=50, min_length=20, do_sample=False)[0]['summary_text']
            st.success("✅ Summarization Completed!")
            st.text_area("📌 **Summarized English Text:**", summary, height=150)
        except Exception as e:
            st.error(f"❌ Summarization Failed: {e}")
            summary = None

    # Step 4: Sentiment Analysis
    if summary:
        st.write("😊 **Step 4: Sentiment Analysis...**")
        try:
            sentiment_results = sentiment_analyzer(summary)
            sentiment_label = sentiment_results[0]['label']
            sentiment_score = sentiment_results[0]['score']

            # Categorizing sentiment into Positive, Negative, and Neutral
            if "1 star" in sentiment_label or "2 stars" in sentiment_label:
                sentiment_category = "NEGATIVE"
            elif "3 stars" in sentiment_label:
                sentiment_category = "NEUTRAL"
            else:
                sentiment_category = "POSITIVE"

            st.success(f"✅ Sentiment: {sentiment_category} ({sentiment_score:.2f})")

            # Sentiment Visualization
            labels = ["Positive", "Negative", "Neutral"]
            scores = [
                sentiment_score if sentiment_category == "POSITIVE" else 0,
                sentiment_score if sentiment_category == "NEGATIVE" else 0,
                sentiment_score if sentiment_category == "NEUTRAL" else 0.5
            ]

            fig, ax = plt.subplots()
            ax.bar(labels, scores, color=['green', 'red', 'gray'])
            ax.set_ylabel("Confidence Score")
            ax.set_ylim(0, 1)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Sentiment Analysis Failed: {e}")

    # Cleanup temp file
    os.remove(file_path)

st.info("👆 Upload a Hindi audio file to start the process!")
