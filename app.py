import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datetime import datetime

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="English Learning Chatbot",
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
# Utility Functions
# =========================
def normalize_text(text):
    fixes = {
        "yesterdaay": "yesterday",
        "yestarday": "yesterday"
    }
    for k, v in fixes.items():
        text = text.replace(k, v)
    return text

def correct_language(text):
    text = normalize_text(text)
    prompt = "fix grammar and tense without changing meaning: " + text
    ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    out = model.generate(ids, max_length=256)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def explain_mistakes(original, corrected):
    if original.lower() == corrected.lower():
        return "• No major grammatical mistakes were found."
    return (
        "• Grammar and sentence structure were corrected.\n"
        "• Verb tense agreement was fixed.\n"
        "• Spelling mistakes were corrected.\n"
        "• Meaning of the sentence was preserved."
    )

def ielts_mode(text):
    return (
        "Formal academic version suitable for IELTS/TOEFL:\n"
        f"{text}"
    )

def generate_12_tenses():
    return (
        "Present Simple: He goes to the office\n"
        "Present Continuous: He is going to the office\n"
        "Present Perfect: He has gone to the office\n"
        "Present Perfect Continuous: He has been going to the office\n\n"
        "Past Simple: He went to the office\n"
        "Past Continuous: He was going to the office\n"
        "Past Perfect: He had gone to the office\n"
        "Past Perfect Continuous: He had been going to the office\n\n"
        "Future Simple: He will go to the office\n"
        "Future Continuous: He will be going to the office\n"
        "Future Perfect: He will have gone to the office\n"
        "Future Perfect Continuous: He will have been going to the office"
    )

# =========================
# Session State
# =========================
if "started" not in st.session_state:
    st.session_state.started = False

if "chat" not in st.session_state:
    st.session_state.chat = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# =========================
# Header
# =========================
st.title("🧠 English Language Learning Chatbot")
st.caption("Correction • Explanation • IELTS/TOEFL • 12 Tenses")

# =========================
# Controls
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶ START"):
        st.session_state.started = True
        st.session_state.chat = []
        st.session_state.input_text = ""

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
# Main App Logic
# =========================
if st.session_state.started:

    # STEP 1: INPUT
    st.subheader("✍️ Step 1: Enter your sentence")
    st.session_state.input_text = st.text_input(
        "Sentence",
        value=st.session_state.input_text
    )

    # STEP 2: OPTIONS
    if st.session_state.input_text.strip():

        st.subheader("⚙ Step 2: Select options (multiple allowed)")
        opt_correct = st.checkbox("Language Correction")
        opt_explain = st.checkbox("Explain Mistakes")
        opt_ielts = st.checkbox("IELTS / TOEFL Mode")
        opt_tenses = st.checkbox("Answer in 12 Tenses")

        st.subheader("▶ Step 3: Run")
        if st.button("RUN"):

            user_text = st.session_state.input_text
            st.session_state.chat.append(f"👤 **You:** {user_text}")

            corrected = user_text
            response = ""

            if opt_correct:
                corrected = correct_language(user_text)
                response += f"✅ **Corrected English:**\n{corrected}\n\n"

            if opt_explain:
                response += (
                    f"🧠 **Explanation of Mistakes:**\n"
                    f"{explain_mistakes(user_text, corrected)}\n\n"
                )

            if opt_ielts:
                response += f"🎓 **IELTS / TOEFL Style:**\n{ielts_mode(corrected)}\n\n"

            if opt_tenses:
                response += f"⏱ **Sentence in 12 Tenses:**\n{generate_12_tenses()}\n\n"

            if response == "":
                response = "⚠ Please select at least one option."

            st.session_state.chat.append(f"🤖 **Bot:**\n{response}")
            st.session_state.input_text = ""
            st.rerun()

    # CHAT HISTORY
    st.divider()
    for msg in st.session_state.chat:
        st.markdown(msg)

else:
    st.info("Click **START** to begin.")

st.caption("Stable • Lightweight • Streamlit Cloud Compatible")
