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
# GPT Helper
# =========================
def gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert English teacher."},
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
st.caption("Accurate • Meaning-Preserving • Exam-Ready")

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

        st.subheader("⚙ Step 2: Select options")
        opt_correct = st.checkbox("Language Correction")
        opt_explain = st.checkbox("Explain Mistakes")
        opt_ielts = st.checkbox("IELTS / TOEFL Mode")
        opt_tenses = st.checkbox("Answer in 12 Tenses")

        if st.button("▶ RUN"):

            user_text = st.session_state.input_text
            st.session_state.chat.append(f"👤 **You:** {user_text}")

            output = ""

            if opt_correct:
                output += "✅ **Corrected English:**\n"
                output += gpt(
                    f"Correct the grammar and tense of this sentence without changing its meaning:\n{user_text}"
                ) + "\n\n"

            if opt_explain:
                output += "🧠 **Explanation of Mistakes:**\n"
                output += gpt(
                    f"Explain the grammar and tense mistakes in simple bullet points:\n{user_text}"
                ) + "\n\n"

            if opt_ielts:
                output += "🎓 **IELTS / TOEFL Version:**\n"
                output += gpt(
                    f"Rewrite this sentence in a formal academic IELTS/TOEFL style:\n{user_text}"
                ) + "\n\n"

            if opt_tenses:
                output += "⏱ **Sentence in 12 Tenses:**\n"
                output += gpt(
                    f"Write this sentence correctly in all 12 English tenses:\n{user_text}"
                ) + "\n\n"

            if not output:
                output = "⚠ Please select at least one option."

            st.session_state.chat.append(f"🤖 **Bot:**\n{output}")
            st.session_state.input_text = ""
            st.rerun()

    st.divider()
    for msg in st.session_state.chat:
        st.markdown(msg)

else:
    st.info("Click **START** to begin.")

st.caption("Powered by GPT • Linguistically Correct • Production-Grade")
