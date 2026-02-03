import streamlit as st
import os
from openai import OpenAI
from datetime import datetime

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="English GPT Tutor",
    page_icon="🧠",
    layout="centered"
)

# =========================
# OpenAI Client
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# GPT – SINGLE CALL ONLY
# =========================
def gpt_all(text, do_correct, do_explain, do_ielts, do_tenses):
    tasks = []
    if do_correct:
        tasks.append("1. Correct the grammar and tense without changing meaning.")
    if do_explain:
        tasks.append("2. Explain the mistakes briefly in bullet points.")
    if do_ielts:
        tasks.append("3. Rewrite in formal IELTS/TOEFL academic style.")
    if do_tenses:
        tasks.append("4. Rewrite the sentence correctly in all 12 English tenses.")

    if not tasks:
        return "⚠ Please select at least one option."

    prompt = f"""
You are an expert English teacher.

Sentence:
{text}

Tasks:
{chr(10).join(tasks)}

Rules:
- Do NOT change the meaning or time reference.
- Be grammatically correct.
- For 12 tenses, use proper verb forms (not labels).
- Keep output clear and structured.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional English language instructor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

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
st.title("🧠 English GPT Tutor")
st.caption("Accurate • IELTS Ready • Meaning Preserved • Rate-Limit Safe")

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
# Main App
# =========================
if st.session_state.started:

    st.subheader("✍️ Step 1: Enter sentence")
    st.session_state.input_text = st.text_input(
        "Sentence",
        value=st.session_state.input_text
    )

    if st.session_state.input_text.strip():

        st.subheader("⚙ Step 2: Select options (multiple allowed)")
        opt_correct = st.checkbox("Language Correction")
        opt_explain = st.checkbox("Explain Mistakes")
        opt_ielts = st.checkbox("IELTS / TOEFL Mode")
        opt_tenses = st.checkbox("Answer in 12 Tenses")

        if st.button("▶ RUN"):

            user_text = st.session_state.input_text
            st.session_state.chat.append(f"👤 **You:** {user_text}")

            try:
                result = gpt_all(
                    user_text,
                    opt_correct,
                    opt_explain,
                    opt_ielts,
                    opt_tenses
                )
            except Exception as e:
                result = "⚠ API limit reached. Please wait a few seconds and try again."

            st.session_state.chat.append(f"🤖 **Bot:**\n{result}")
            st.session_state.input_text = ""
            st.rerun()

    st.divider()
    for msg in st.session_state.chat:
        st.markdown(msg)

else:
    st.info("Click **START** to begin.")

st.caption("Single-call GPT • Stable • Cost-efficient")
