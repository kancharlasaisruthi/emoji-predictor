import streamlit as st 
import numpy as np
import pickle
import json
import torch
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sentence_transformers import SentenceTransformer
# Load trained LSTM model
model = load_model('LSTM.h5')


tokenizer = SentenceTransformer("all-MiniLM-L6-v2")



# Load emoji mapping from JSON
with open('emoji_map.json', 'r', encoding='utf-8') as f:
    emoji_data = json.load(f)
emoji_dict = {int(k): v[0] for k, v in emoji_data.items()}

def prdict_emoji(model,tokenizer,text):
    token_list=tokenizer.encode([text])
    token_list=token_list.reshape(token_list.shape[0],1,token_list.shape[1])
    predicted=model.predict(token_list,verbose=0)
    predicted_word_index=np.argmax(predicted,axis=1)  ## This line means: which word have high probability take that index

    return predicted_word_index

# Streamlit UI
st.set_page_config(page_title="Emoji Predictor", page_icon="😊", layout="centered")
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f1f2f6, #dff9fb);
    }

    .main-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        margin: 2rem auto;
        max-width: 600px;
    }

    .title {
        text-align: center;
        color: #2d3436;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        text-align: center;
        color: #636e72;
        margin-bottom: 2rem;
    }

    .result-box {
        background: #fdfefe;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
        border-left: 5px solid #55efc4;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }

    .big-emoji {
        font-size: 4rem;
        margin: 1rem 0;
    }

    .stButton button {
        background: linear-gradient(45deg, #81ecec, #74b9ff);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        width: 100%;
        transition: transform 0.2s ease;
    }

    .stButton button:hover {
        transform: translateY(-2px);
    }

    .stTextInput input {
        border-radius: 25px;
        border: 2px solid #dcdde1;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        background: #ffffff;
    }

    .stTextInput input:focus {
        border-color: #74b9ff;
        box-shadow: 0 0 10px rgba(116, 185, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">😊 Emoji Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Type a message and I\'ll guess the perfect emoji!</p>', unsafe_allow_html=True)
input_text = st.text_input("Your message:", placeholder="I'm feeling happy today...")

if st.button("🎯 Predict Emoji"):
    if input_text.strip():
        with st.spinner("Thinking..."):
            emoji_index = prdict_emoji(model, tokenizer, input_text)
            predicted_emoji = emoji_dict.get(emoji_index, "❓")
        
        st.markdown(f"""
        <div class="result-box">
            <div style="color: #636e72; font-size: 1.1rem;">Perfect emoji for:</div>
            <div style="color: #2d3436; font-style: italic; margin: 0.5rem 0;">"{input_text}"</div>
            <div class="big-emoji">{predicted_emoji}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please enter a message!")

st.markdown("---")
st.markdown("**💡 Try these examples:**")
col1, col2 = st.columns(2)
with col1:
    st.markdown("• I love pizza")
    st.markdown("• Feeling sad today")
with col2:
    st.markdown("• So excited!")
    st.markdown("• Good morning")
