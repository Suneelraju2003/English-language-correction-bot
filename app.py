import streamlit as st
import torch
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM
)

st.set_page_config(page_title="English Correction & Translator", page_icon="🧠")

# =========================
# Load Models (Cached)
# =========================
@st.cache_resource
def load_models():
    # Grammar model
    grammar_model_name = "vennify/t5-base-grammar-correction"
    g_tokenizer = T5Tokenizer.from_pretrained(grammar_model_name)
    g_model = T5ForConditionalGeneration.from_pretrained(grammar_model_name)

    # English → Hindi
    hi_model_name = "Helsinki-NLP/opus-mt-en-hi"
    hi_tokenizer = AutoTokenizer.from_pretrained(hi_model_name)
    hi_model = AutoModelForSeq2SeqLM.from_pretrained(hi_model_name)

    # English → Telugu (IndicTrans2)
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

# =========================
# Functions
# =========================
def correct_grammar(text):
    inp = "grammar: " + text
    ids = grammar_tokenizer.encode(inp, return_tensors="pt", truncation=True)
    out = grammar_model.generate(ids, max_length=256)
    return grammar_tokenizer.decode(out[0], skip_special_tokens=True)

def translate(text, tokenizer, model, tgt_lang=None):
    if tgt_lang:
        text = f"<2{tgt_lang}> {text}"
    ids = tokenizer(text, return_tensors="pt", truncation=True)
    out = model.generate(**ids, max_length=256)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# =========================
# UI
# =========================
st.title("🧠 English Correction & Translation Bot")

options = st.multiselect(
    "Select operations (multiple allowed):",
    [
        "Grammar & Tense Correction",
        "Improve Vocabulary",
        "Translate to Hindi",
        "Translate to Telugu"
    ]
)

text = st.text_area("✍️ Enter English text")

if st.button("▶ Run"):
    if not text.strip():
        st.warning("Please enter text.")
    else:
        with st.spinner("Processing..."):
            corrected = text

            if "Grammar & Tense Correction" in options or "Improve Vocabulary" in options:
                corrected = correct_grammar(text)
                st.subheader("✅ Corrected English")
                st.write(corrected)

            if "Translate to Hindi" in options:
                hi = translate(corrected, hi_tokenizer, hi_model)
                st.subheader("🇮🇳 Hindi")
                st.write(hi)

            if "Translate to Telugu" in options:
                te = translate(corrected, te_tokenizer, te_model, tgt_lang="te")
                st.subheader("🇮🇳 Telugu")
                st.write(te)

st.caption("100% Free • Open Source • No API Keys")
