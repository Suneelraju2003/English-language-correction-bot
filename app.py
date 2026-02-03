import streamlit as st
import torch
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    MarianTokenizer, MarianMTModel
)
from datetime import datetime

# ======================
# Page config
# ======================
st.set_page_config(
    page_title="English AI Chatbot",
    page_icon="🧠",
    layout="centered"
)

# ======================
# Load models (LIGHT ONLY)
# ======================
@st.cache_resource
def load_models():
    # Grammar
    g_name = "vennify/t5-base-grammar-correction"
    g_tok = T5Tokenizer.from_pretrained(g_name)
    g_mod = T5ForConditionalGeneration.from_pretrained(g_name)

    # English → Hindi
    hi_name = "Helsinki-NLP/opus-mt-en-hi"
    hi_tok = MarianTokenizer.from_pretrained(hi_name)
    hi_mod = MarianMTModel.from_pretrained(hi_name)

    # English → Telugu (light Marian model)
    te_name = "Helsinki-NLP/opus-mt-en-te"
    te_tok = MarianTokenizer.from_pretrained(te_name)
    te_mod = MarianMTModel.from_pretrained(te_name)

    return g_tok, g_mod, hi_tok, hi_mod, te_tok, te_mod

g_tok, g_mod, hi_tok, hi_mod, te_tok, te_mod = load_models()

# ======================
# Functions
# ======================
def correct_grammar(text):
    ids = g_tok.encode("grammar: " + text, return_tensors="pt", truncation=True)
    out = g_mod.generate(ids, max_length=256)
    return g_tok.decode(out[0], skip_special_tokens=True)

def translate(text, tok, mod):
    ids = tok(text, return_tensors="pt", truncation=True)
    out = mod.generate(**ids, max_length=256)
    return tok.decode(out[0], skip_special_tokens=True)

# ======================
# Session state
# ======================
if "started" not in st.session_state:
    st.session_state.started = False

if "chat" not in st.session_state:
    st.session_state.chat = []

# ======================
# UI
# ======================
st.title("🧠 English AI Chatbot")
st.caption("Chat • Grammar • Translation")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶ START"):
        st.session_state.started = True
        st.session_state.chat = []

with col2:
    if st.button("⏹ STOP"):
        st.session_state.started = False

with col3:
    if st.session_state.chat:
        st.download_button(
            "⬇ DOWNLOAD",
            "\n\n".join(st.session_state.chat),
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        )

st.divider()

if st.session_state.started:
    st.subheader("⚙ Options")
    opt_grammar = st.checkbox("Grammar & Tense")
    opt_hi = st.checkbox("English → Hindi")
    opt_te = st.checkbox("English → Telugu")

    st.divider()

    for msg in st.session_state.chat:
        st.markdown(msg)

    user_text = st.chat_input("Type your English sentence...")

    if user_text:
        st.session_state.chat.append(f"👤 **You:** {user_text}")

        reply = ""
        corrected = user_text

        if opt_grammar:
            corrected = correct_grammar(user_text)
            reply += f"✅ **Corrected:**\n{corrected}\n\n"

        if opt_hi:
            reply += f"🇮🇳 **Hindi:**\n{translate(corrected, hi_tok, hi_mod)}\n\n"

        if opt_te:
            reply += f"🇮🇳 **Telugu:**\n{translate(corrected, te_tok, te_mod)}\n\n"

        if not reply:
            reply = "⚠ Please select at least one option."

        st.session_state.chat.append(f"🤖 **Bot:**\n{reply}")
        st.rerun()

else:
    st.info("Click **START** to begin chatting.")

st.caption("Optimized for Streamlit Cloud")
