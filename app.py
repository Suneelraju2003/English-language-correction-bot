from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters
)
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# =========================
# Load Models (once)
# =========================
GRAMMAR_MODEL = "vennify/t5-base-grammar-correction"
HI_MODEL = "Helsinki-NLP/opus-mt-en-hi"
TE_MODEL = "ai4bharat/indictrans2-en-indic-1B"

grammar_tokenizer = T5Tokenizer.from_pretrained(GRAMMAR_MODEL)
grammar_model = T5ForConditionalGeneration.from_pretrained(GRAMMAR_MODEL)

hi_tokenizer = AutoTokenizer.from_pretrained(HI_MODEL)
hi_model = AutoModelForSeq2SeqLM.from_pretrained(HI_MODEL)

te_tokenizer = AutoTokenizer.from_pretrained(TE_MODEL, trust_remote_code=True)
te_model = AutoModelForSeq2SeqLM.from_pretrained(TE_MODEL, trust_remote_code=True)

# =========================
# Helper Functions
# =========================
def correct_grammar(text):
    text = "grammar: " + text
    ids = grammar_tokenizer.encode(text, return_tensors="pt", truncation=True)
    out = grammar_model.generate(ids, max_length=256)
    return grammar_tokenizer.decode(out[0], skip_special_tokens=True)

def translate_hindi(text):
    ids = hi_tokenizer(text, return_tensors="pt", truncation=True)
    out = hi_model.generate(**ids, max_length=256)
    return hi_tokenizer.decode(out[0], skip_special_tokens=True)

def translate_telugu(text):
    text = "<2te> " + text
    ids = te_tokenizer(text, return_tensors="pt", truncation=True)
    out = te_model.generate(**ids, max_length=256)
    return te_tokenizer.decode(out[0], skip_special_tokens=True)

# =========================
# /start Command
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    context.user_data["options"] = set()

    keyboard = [
        [InlineKeyboardButton("✅ Grammar Correction", callback_data="grammar")],
        [InlineKeyboardButton("🌐 English → Hindi", callback_data="hi")],
        [InlineKeyboardButton("🌐 English → Telugu", callback_data="te")],
        [InlineKeyboardButton("▶ RUN", callback_data="run")]
    ]

    await update.message.reply_text(
        "🧠 *English Correction Bot*\n\n"
        "STEP 1️⃣ Select one or more options\n"
        "STEP 2️⃣ Click ▶ RUN\n"
        "STEP 3️⃣ Send your English sentence",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

# =========================
# Button Handler
# =========================
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    options = context.user_data.get("options", set())
    choice = query.data

    if choice == "run":
        if not options:
            await query.edit_message_text("⚠️ Please select at least one option.")
            return

        context.user_data["ready"] = True
        await query.edit_message_text(
            "✍️ *Now send your English sentence*",
            parse_mode="Markdown"
        )
        return

    options.add(choice)
    context.user_data["options"] = options

    selected = "\n".join(f"• {o}" for o in options)

    await query.edit_message_text(
        "🧠 *English Correction Bot*\n\n"
        "STEP 1️⃣ Select options\n"
        "STEP 2️⃣ Click ▶ RUN\n"
        "STEP 3️⃣ Send sentence\n\n"
        "✅ *Selected options:*\n"
        f"{selected}",
        parse_mode="Markdown",
        reply_markup=query.message.reply_markup
    )

# =========================
# Text Handler
# =========================
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("ready"):
        await update.message.reply_text("⚠️ Use /start and select options first.")
        return

    text = update.message.text
    options = context.user_data["options"]

    result = f"❌ Original:\n{text}\n\n"
    corrected = text

    if "grammar" in options:
        corrected = correct_grammar(text)
        result += f"✅ Corrected:\n{corrected}\n\n"

    if "hi" in options:
        result += f"🇮🇳 Hindi:\n{translate_hindi(corrected)}\n\n"

    if "te" in options:
        result += f"🇮🇳 Telugu:\n{translate_telugu(corrected)}\n\n"

    await update.message.reply_text(result)
    context.user_data.clear()

# =========================
# Main
# =========================
def main():
    BOT_TOKEN = "8500153960:AAFJre6HR8kguZ5XA5ALP1E2UoSaxmQKZKM"

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("🤖 Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
