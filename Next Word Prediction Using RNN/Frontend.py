import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import os

# -------------------------------
# 🎨 Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="Next Word Prediction", page_icon="🧩", layout="centered")

# Custom CSS for better UI
st.markdown("""
    <style>
        body {
            background-color: #0f172a;
            color: white;
        }
        .stApp {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        }
        .title {
            text-align: center;
            color: #38bdf8;
            font-family: 'Trebuchet MS', sans-serif;
            font-size: 45px;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            color: #f8fafc;
            font-size: 18px;
            margin-bottom: 30px;
        }
        .stTextInput > div > div > input {
            background-color: #1e293b;
            color: white;
        }
        .result-box {
            background-color: #1e293b;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-top: 20px;
            color: #38bdf8;
            font-size: 20px;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            color: #94a3b8;
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 🧩 Load Model & Tokenizer
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("nwp.h5")
    return model

model = load_model()

@st.cache_resource
def load_tokenizer():
    # Assuming tokenizer was created in same script earlier
    import tensorflow as tf
    import pandas as pd
    df = pd.read_csv("D:\\Data Science Daily Updates\\Senapthi_calss_Notes\\DEEP LEARNING CLASS\\5.RNN\\Next_Word_Prediction\\tmdb_5000_movies.csv")
    df = df['original_title']
    movie_name = df.to_list()
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(movie_name)
    vocab_array = np.array(list(tokenizer.word_index.keys()))
    return tokenizer, vocab_array

tokenizer, vocab_array = load_tokenizer()


# -------------------------------
# 🔮 Prediction Function
# -------------------------------
def make_prediction(text, n_words):
    for i in range(n_words):
        text_tokenize = tokenizer.texts_to_sequences([text])
        text_padded = tf.keras.preprocessing.sequence.pad_sequences(text_tokenize, maxlen=14)
        prediction = np.squeeze(np.argmax(model.predict(text_padded), axis=-1))
        prediction = str(vocab_array[prediction - 1])
        text += " " + prediction
    return text


# -------------------------------
# 🖥️ Frontend Layout
# -------------------------------
st.markdown("<div class='title'>Next Word Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Powered by TensorFlow • LSTM Seq2Seq Model</div>", unsafe_allow_html=True)

# Input from user
text_input = st.text_input("Enter a starting word or phrase:", placeholder="Type something like 'cloudy' or 'life of'")

num_words = st.slider("Number of words to predict:", 1, 20, 5)

# Predict button
if st.button("🔮 Predict Next Words"):
    if text_input.strip() == "":
        st.warning("Please enter a word or phrase before predicting!")
    else:
        with st.spinner("Generating next words..."):
            prediction = make_prediction(text_input, num_words)
        st.markdown(f"<div class='result-box'>✨ Predicted Text:<br><b>{prediction}</b></div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>Developed by Aneelkumar Muppana • Deep Learning Project 🧠</div>", unsafe_allow_html=True)
