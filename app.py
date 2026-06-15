import re
import json
import joblib
import streamlit as st
from pathlib import Path

try:
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
except Exception:
    StopWordRemoverFactory = None


st.set_page_config(
    page_title="Analisis Sentimen DANA",
    page_icon="💬",
    layout="centered"
)


def load_first_existing(paths):
    """
    Mencari file dari beberapa kemungkinan lokasi.
    Dipakai supaya app tidak error kalau file ada di root atau di folder models.
    """
    for path in paths:
        if Path(path).exists():
            return path
    return None


@st.cache_resource
def load_model():
    model_path = load_first_existing([
        "best_model.pkl",
        "models/best_model.pkl",
        "model/best_model.pkl"
    ])

    if model_path is None:
        st.error(
            "File model tidak ditemukan. Pastikan file `best_model.pkl` sudah diupload ke GitHub."
        )
        st.stop()

    model = joblib.load(model_path)

    metadata_path = load_first_existing([
        "metadata.json",
        "models/metadata.json",
        "model/metadata.json"
    ])

    if metadata_path is not None:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "best_model": "Model Terbaik",
            "selection_metric": "F1-Score / Accuracy",
            "labels": ["negatif", "netral", "positif"]
        }

    return model, metadata


model, metadata = load_model()


# Stopword dan normalisasi dibuat sama seperti notebook training
@st.cache_resource
def load_stopwords():
    stopword_path = load_first_existing([
        "stopwords_indonesia.pkl",
        "models/stopwords_indonesia.pkl",
        "model/stopwords_indonesia.pkl"
    ])

    if stopword_path is not None:
        try:
            return set(joblib.load(stopword_path))
        except Exception:
            pass

    if StopWordRemoverFactory is not None:
        factory = StopWordRemoverFactory()
        return set(factory.get_stop_words())

    return {
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "atau",
        "pada", "adalah", "saya", "aku", "kamu", "dia", "mereka", "kami", "kita"
    }


stopwords_indonesia = load_stopwords()

negation_words = {
    "tidak", "bukan", "jangan", "belum", "kurang", "tanpa",
    "ga", "gak", "nggak", "ngga"
}

stopwords_indonesia = stopwords_indonesia - negation_words

custom_stopwords = {
    "dana", "aplikasi", "apk", "app", "playstore", "play", "store",
    "nya", "nih", "dong", "sih", "deh", "lah", "ya", "aja",
    "kak", "min", "admin"
}

all_stopwords = stopwords_indonesia.union(custom_stopwords)

normalization_dict = {
    "gak": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "gk": "tidak",
    "tdk": "tidak",
    "tak": "tidak",
    "yg": "yang",
    "dgn": "dengan",
    "utk": "untuk",
    "dlm": "dalam",
    "sm": "sama",
    "sma": "sama",
    "tp": "tapi",
    "tpi": "tapi",
    "krn": "karena",
    "dr": "dari",
    "blm": "belum",
    "sdh": "sudah",
    "udh": "sudah",
    "udah": "sudah",
    "bgt": "banget",
    "bgtt": "banget",
    "mantab": "mantap",
    "mantapp": "mantap",
    "lemot": "lambat",
    "eror": "error",
    "err": "error",
    "gabisa": "tidak bisa",
    "topup": "top up",
    "tf": "transfer"
}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"#[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [normalization_dict.get(token, token) for token in tokens]
    tokens = [token for token in tokens if token not in all_stopwords and len(token) > 2]

    return " ".join(tokens)


st.title("Analisis Sentimen Review Aplikasi DANA")

st.write(
    "Aplikasi ini digunakan untuk memprediksi sentimen review pengguna aplikasi DANA "
    "menggunakan model machine learning terbaik dari hasil evaluasi."
)

st.info(f"Model yang digunakan: {metadata.get('best_model', 'Model Terbaik')}")

user_input = st.text_area(
    "Masukkan review aplikasi DANA:",
    placeholder="Contoh: Aplikasi DANA sering error saat transfer saldo."
)

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Masukkan review terlebih dahulu.")
    else:
        cleaned = clean_text(user_input)

        try:
            prediction = model.predict([cleaned])[0]

            st.subheader("Hasil Prediksi")
            st.success(f"Sentimen: {str(prediction).upper()}")

            with st.expander("Lihat hasil preprocessing"):
                st.write(cleaned)

        except Exception as e:
            st.error("Terjadi error saat melakukan prediksi.")
            st.write("Kemungkinan model belum tersimpan sebagai Pipeline TF-IDF + Classifier.")
            st.code(str(e))
