import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datetime import datetime

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="English Correction Chat",
    page_icon="🧠",
    layout="centered"
)

# =========================
# Load Model (LIGHT)
# =========================
@st.cache_resource
def load_model():
    model_name = "vennify/t5-base-grammar-correction"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# =========================
# Grammar Function
# =========================
def correct_english(text):
    input_text = "grammar: " + text
    input_ids = tokenizer.encode(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    outputs = model.generate(input_ids, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================
# Session State
# =========================
if "started" not in st.session_state:
    st.session_state.started = False

if "chat" not in st.session_state:
    st.session_state.chat = []

# =========================
# UI Header
# =========================
st.title("🧠 English Correction Chatbot")
st.caption("Grammar • Tense • Vocabulary (Lightweight)")

# =========================
# Controls
# =========================
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

# =========================
# Chat Area
# =========================
if st.session_state.started:

    for msg in st.session_state.chat:
        st.markdown(msg)

    user_input = st.chat_input("Type your English sentence...")

    if user_input:
        st.session_state.chat.append(f"👤 **You:** {user_input}")

        corrected = correct_english(user_input)

        st.session_state.chat.append(
            f"🤖 **Bot:**\n✅ **Corrected English:**\n{corrected}"
        )

        st.rerun()

else:
    st.info("Click **START** to begin chatting.")

st.caption("Optimized for Streamlit Cloud Free Tier")
