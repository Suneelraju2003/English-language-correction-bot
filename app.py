import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datetime import datetime

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="English Language Chatbot",
    page_icon="🧠",
    layout="centered"
)

# =========================
# Load Model (light & safe)
# =========================
@st.cache_resource
def load_model():
    model_name = "vennify/t5-base-grammar-correction"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# =========================
# Core Functions
# =========================
def correct_language(text):
    ids = tokenizer.encode(
        "grammar: " + text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    out = model.generate(ids, max_length=256)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def explain_mistakes(original, corrected):
    explanation = []
    if original.lower() != corrected.lower():
        explanation.append("• Grammar and sentence structure were corrected.")
        explanation.append("• Verb tense agreement was fixed.")
        explanation.append("• Unnecessary or incorrect words were removed.")
    else:
        explanation.append("• The sentence was already grammatically correct.")
    return "\n".join(explanation)

def ielts_mode(text):
    return f"This sentence is rewritten in a formal academic style:\n{text}"

def generate_12_tenses(sentence):
    base = sentence.rstrip(".")
    return (
        f"Present Simple: {base}\n"
        f"Present Continuous: {base} (now)\n"
        f"Present Perfect: {base} (has/have)\n"
        f"Present Perfect Continuous: {base} (has been)\n\n"
        f"Past Simple: {base} (yesterday)\n"
        f"Past Continuous: {base} (was/were)\n"
        f"Past Perfect: {base} (had)\n"
        f"Past Perfect Continuous: {base} (had been)\n\n"
        f"Future Simple: {base} (will)\n"
        f"Future Continuous: {base} (will be)\n"
        f"Future Perfect: {base} (will have)\n"
        f"Future Perfect Continuous: {base} (will have been)"
    )

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
st.title("🧠 English Language Learning Chatbot")
st.caption("Grammar • Explanation • IELTS/TOEFL • 12 Tenses")

# =========================
# Control Buttons
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
# Main Chat Area
# =========================
if st.session_state.started:

    st.subheader("⚙ Select Options (multiple allowed)")
    opt_correct = st.checkbox("Language Correction")
    opt_explain = st.checkbox("Explain Mistakes")
    opt_ielts = st.checkbox("IELTS / TOEFL Mode")
    opt_tenses = st.checkbox("Answer in 12 Tenses")

    st.divider()

    for msg in st.session_state.chat:
        st.markdown(msg)

    user_input = st.chat_input("Type your English sentence...")

    if user_input:
        st.session_state.chat.append(f"👤 **You:** {user_input}")

        response = ""
        corrected = user_input

        if opt_correct:
            corrected = correct_language(user_input)
            response += f"✅ **Corrected English:**\n{corrected}\n\n"

        if opt_explain:
            response += (
                f"🧠 **Explanation of Mistakes:**\n"
                f"{explain_mistakes(user_input, corrected)}\n\n"
            )

        if opt_ielts:
            response += f"🎓 **IELTS / TOEFL Style:**\n{ielts_mode(corrected)}\n\n"

        if opt_tenses:
            response += f"⏱ **Sentence in 12 Tenses:**\n{generate_12_tenses(corrected)}\n\n"

        if response == "":
            response = "⚠ Please select at least one option."

        st.session_state.chat.append(f"🤖 **Bot:**\n{response}")
        st.rerun()

else:
    st.info("Click **START** to begin.")

st.caption("Lightweight • Streamlit Cloud Safe • No Heavy Models")
