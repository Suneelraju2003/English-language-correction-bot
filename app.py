import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# -----------------------------
# Load Model and Tokenizer
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "vennify/t5-base-grammar-correction"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# Grammar Correction Function
# -----------------------------
def correct_english(text):
    input_text = "grammar: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )

    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="English Correction Chatbot", page_icon="🧠")

st.title("🧠 English Grammar & Tense Correction Chatbot")
st.write("Type any English sentence. The bot will correct grammar, tense, and vocabulary.")

user_input = st.text_area("✍️ Enter your sentence:", height=120)

if st.button("✅ Correct English"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        with st.spinner("Correcting..."):
            corrected = correct_english(user_input)
        st.success("✔️ Corrected Sentence:")
        st.write(corrected)

st.markdown("---")
st.caption("Free • Offline • Open Source • No API keys")
