import streamlit as st
import torch
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
from datetime import datetime

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="English AI Chatbot",
    page_icon="🧠",
    layout="centered"
)

# ===============================
# Load Models (cached)
# ===============================
@st.cache_resource
def load_models():
    # Grammar correction
    g_model_name = "vennify/t5-base-grammar-correction"
    g_tokenizer = T5Tokenizer.from_pretrained(g_model_name)
    g_model = T5ForConditionalGeneration.from_pretrained(g_model_name)

    # English → Hindi
    hi_model_name = "Helsinki-NLP/opus-mt-en-hi"
    hi_tokenizer = AutoTokenizer.from_pretrained(hi_model_name)
    hi_model = AutoModelForSeq2SeqLM.from_pretrained(hi_model_name)

    # English → Telugu
    te_model_name = "ai4bharat/indictrans2-en-indic-1B"
    te_tokenizer = AutoTokenizer.from_pretrained(te_model_name, trust_remote_code=True)
    te_model = AutoModelForSeq2SeqLM.from_pretrained(te_model_name, trust_remote_code=True)

    return g_tokenizer, g_model, hi_tokenizer, hi_model, te_tokenizer, te_model

(
    grammar_tokenizer,
    grammar_model,
    hi_tokenizer,
    hi_model,
    te_tokenizer,
    te_model
) = load_models()

# ===============================
# Functions
# ===============================
def correct_grammar(text):
    inp = "grammar: " + text
    ids = grammar_tokenizer.encode(inp, return_tensors="pt", truncation=True)
    out = grammar_model.generate(ids, max_length=256)
    return grammar_tokenizer.decode(out[0], skip_special_tokens=True)

def translate_hi(text):
    ids = hi_tokenizer(text, return_tensors="pt", truncation=True)
    out = hi_model.generate(**ids, max_length=256)
    return hi_tokenizer.decode(out[0], skip_special_tokens=True)

def translate_te(text):
    text = "<2te> " + text
    ids = te_tokenizer(text, return_tensors="pt", truncation=True)
    out = te_model.generate(**ids, max_length=256)
    return te_tokenizer.decode(out[0], skip_special_tokens=True)

# ===============================
# Session State
# ===============================
if "started" not in st.session_state:
    st.session_state.started = False

if "chat" not in st.session_state:
    st.session_state.chat = []

# ===============================
# UI – Header
# ===============================
st.title("🧠 English AI Chatbot")
st.caption("Grammar • Vocabulary • Translation • Chat-based")

# ===============================
# START / STOP CONTROLS
# ===============================
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶ START CHAT"):
        st.session_state.started = True
        st.session_state.chat = []

with col2:
    if st.button("⏹ STOP CHAT"):
        st.session_state.started = False

with col3:
    if st.session_state.chat:
        chat_text = "\n\n".join(st.session_state.chat)
        st.download_button(
            "⬇ DOWNLOAD CHAT",
            chat_text,
            file_name=f"english_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        )

# ===============================
# OPTIONS (only if started)
# ===============================
if st.session_state.started:
    st.subheader("⚙ Select Options")

    col1, col2, col3 = st.columns(3)
    with col1:
        opt_grammar = st.checkbox("Grammar & Tense")
    with col2:
        opt_hi = st.checkbox("English → Hindi")
    with col3:
        opt_te = st.checkbox("English → Telugu")

    st.divider()

    # ===============================
    # CHAT DISPLAY
    # ===============================
    for msg in st.session_state.chat:
        st.markdown(msg)

    # ===============================
    # CHAT INPUT
    # ===============================
    user_input = st.chat_input("Type your English sentence here...")

    if user_input:
        st.session_state.chat.append(f"👤 **You:** {user_input}")

        response = ""

        corrected = user_input
        if opt_grammar:
            corrected = correct_grammar(user_input)
            response += f"✅ **Corrected English:**\n{corrected}\n\n"

        if opt_hi:
            response += f"🇮🇳 **Hindi:**\n{translate_hi(corrected)}\n\n"

        if opt_te:
            response += f"🇮🇳 **Telugu:**\n{translate_te(corrected)}\n\n"

        if response == "":
            response = "⚠ Please select at least one option."

        st.session_state.chat.append(f"🤖 **Bot:**\n{response}")
        st.rerun()

else:
    st.info("Click **START CHAT** to begin.")

st.caption("100% Free • Open Source • Runs on Streamlit")
