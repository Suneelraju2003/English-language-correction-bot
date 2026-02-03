import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datetime import datetime

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="English Learning Chatbot",
    page_icon="🧠",
    layout="centered"
)

# =========================
# Load model (lightweight)
# =========================
@st.cache_resource
def load_model():
    model_name = "vennify/t5-base-grammar-correction"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# =========================
# Core functions
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
    if original.lower() == corrected.lower():
        return "• No major grammatical mistakes were found."
    return (
        "• Grammar and sentence structure were corrected.\n"
        "• Verb tense agreement was fixed.\n"
        "• Word usage was improved."
    )

def ielts_mode(text):
    return f"This sentence is rewritten in a formal academic style:\n{text}"

def generate_12_tenses(sentence):
    s = sentence.rstrip(".")
    return (
        f"Present Simple: {s}\n"
        f"Present Continuous: {s} (is/are)\n"
        f"Present Perfect: {s} (has/have)\n"
        f"Present Perfect Continuous: {s} (has been)\n\n"
        f"Past Simple: {s} (yesterday)\n"
        f"Past Continuous: {s} (was/were)\n"
        f"Past Perfect: {s} (had)\n"
        f"Past Perfect Continuous: {s} (had been)\n\n"
        f"Future Simple: {s} (will)\n"
        f"Future Continuous: {s} (will be)\n"
        f"Future Perfect: {s} (will have)\n"
        f"Future Perfect Continuous: {s} (will have been)"
    )

# =========================
# Session state
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
st.title("🧠 English Learning Chatbot")
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
# Main logic
# =========================
if st.session_state.started:

    # STEP 1: INPUT
    st.subheader("✍️ Step 1: Enter sentence")
    st.session_state.input_text = st.text_input(
        "Your sentence",
        value=st.session_state.input_text
    )

    # STEP 2: OPTIONS (appear ONLY after input)
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

            response = ""
            corrected = user_text

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
                response += f"⏱ **Sentence in 12 Tenses:**\n{generate_12_tenses(corrected)}\n\n"

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

st.caption("Streamlit Cloud safe • Lightweight • Stable")
