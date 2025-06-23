import os
import pandas as pd
import re
from flask import Flask, request, render_template, session, redirect, url_for
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
openai.api_base = "https://api.groq.com/openai/v1"
openai.api_key = os.getenv("GROQ_API_KEY")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or "super-secret-key"
DATA_PATH = "data/Financial_Data.csv"
df = pd.read_csv(DATA_PATH)


def clean_output(text):
    """Remove unwanted formatting and clean the output"""
    # Remove strike-through formatting (~~text~~)
    text = re.sub(r'~~(.*?)~~', r'\1', text)
    # Remove extra markdown formatting but keep bold for headers
    text = re.sub(r'\*\*(?!Top \d+)(.*?)\*\*', r'\1', text)
    # Convert numbered lists to bullet points
    text = re.sub(r'(\d+)\.\s+([^\n]+)', r'- \2', text)
    # Ensure proper spacing
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


def build_prompt(user_query):
    with open("prompt_template.txt", "r", encoding="utf-8") as file:
        template = file.read()
    column_info = ", ".join(df.columns)
    prompt_text = template.format(columns=column_info)
    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": user_query},
        {"role": "system", "content": (
            "IMPORTANT FORMATTING RULES:\n"
            "1. Never use ~~strike-through~~ text\n"
            "2. Always use vertical bullet points for lists\n"
            "3. Format numbers with $ and commas\n"
            "4. Only bold section headers (like 'which segment had highest profit?:')\n"
            "5. Example format:\n\n"
            "Top 2 product by Profit:\n"
            "- Paseo: $4,797,437.96\n"
            "- VTT: $3,034,608.02"
        )}
    ]


def ask_model(user_query):
    messages = build_prompt(user_query)
    response = openai.ChatCompletion.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.3,
    )
    answer = response.choices[0].message.content.strip()
    return clean_output(answer)


@app.route("/", methods=["GET", "POST"])
def home():
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        user_input = request.form["query"]
        answer = ask_model(user_input)
        session["chat_history"].append({
            "question": user_input,
            "answer": answer
        })
        session.modified = True

    return render_template("index.html", chat_history=session.get("chat_history", []))


@app.route("/new_chat")
def new_chat():
    session.clear()
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)