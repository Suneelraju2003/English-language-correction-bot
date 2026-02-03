from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, CallbackQueryHandler, filters
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    MarianTokenizer, MarianMTModel
)
import torch

# =============================
# Load Models
# =============================
GRAMMAR_MODEL = "vennify/t5-base-grammar-correction"
EN_HI = "Helsinki-NLP/opus-mt-en-hi"
EN_TE = "Helsinki-NLP/opus-mt-en-te"

grammar_tokenizer = T5Tokenizer.from_pretrained(GRAMMAR_MODEL)
grammar_model = T5ForConditionalGeneration.from_pretrained(GRAMMAR_MODEL)

hi_tokenizer = MarianTokenizer.from_pretrained(EN_HI)
hi_model = MarianMTModel.from_pretrained(EN_HI)

te_tokenizer = MarianTokenizer.from_pretrained(EN_TE)
te_model = MarianMTModel.from_pretrained(EN_TE)

# =============================
# Helper Functions
# =============================
def correct_grammar(text):
    input_text = "grammar: " + text
    ids = grammar_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        out = grammar_model.generate(ids, max_length=512, num_beams=5)
    return grammar_tokenizer.decode(out[0], skip_special_tokens=True)

def translate(text, tokenizer, model):
    ids = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        out = model.generate(**ids)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# =============================
# Commands
# =============================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    keyboard = [
        [InlineKeyboardButton("✅ Grammar Correction", callback_data="grammar")],
        [InlineKeyboardButton("🧠 Improve Vocabulary", callback_data="vocab")],
        [InlineKeyboardButton("🧑‍🏫 Explain Mistakes", callback_data="explain")],
        [
            InlineKeyboardButton("🌐 English → Hindi", callback_data="hi"),
            InlineKeyboardButton("🌐 English → Telugu", callback_data="te")
        ],
        [InlineKeyboardButton("▶️ RUN", callback_data="run")]
    ]
    await update.message.reply_text(
        "🧠 English Super Bot\n\n"
        "1️⃣ Type a sentence\n"
        "2️⃣ Select options (multiple allowed)\n"
        "3️⃣ Click RUN",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    choice = query.data
    selected = context.user_data.get("options", set())

    if choice == "run":
        context.user_data["run"] = True
        await query.message.reply_text("✍️ Now send your sentence")
        return

    selected.add(choice)
    context.user_data["options"] = selected
    await query.message.reply_text(f"✔ Selected: {', '.join(selected)}")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    options = context.user_data.get("options", set())

    if not options:
        await update.message.reply_text("⚠️ Please select options first using /start")
        return

    result = f"❌ Original:\n{text}\n\n"

    corrected = text
    if "grammar" in options or "vocab" in options:
        corrected = correct_grammar(text)
        result += f"✅ Corrected:\n{corrected}\n\n"

    if "hi" in options:
        result += f"🇮🇳 Hindi:\n{translate(corrected, hi_tokenizer, hi_model)}\n\n"

    if "te" in options:
        result += f"🇮🇳 Telugu:\n{translate(corrected, te_tokenizer, te_model)}\n\n"

    if "explain" in options:
        result += "🧑‍🏫 Explanation:\nSentence structure, tense, and word usage were corrected."

    await update.message.reply_text(result)
    context.user_data.clear()

# =============================
# Main
# =============================
def main():
    BOT_TOKEN = "8500153960:AAFJre6HR8kguZ5XA5ALP1E2UoSaxmQKZKM"

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("🤖 Bot running...")
    app.run_polling()

if __name__ == "__main__":
    main()
