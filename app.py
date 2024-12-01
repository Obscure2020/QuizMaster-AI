import openai
import os
import re
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables
load_dotenv("static/.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Home route to render the HTML page
@app.route("/")
def home():
    return render_template("index.html")

# Preprocessing text to clean up input
def preprocess_text(text):
    text = re.sub(r'[^\w\s.,?!]', '', text)  # Remove unwanted characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

# API route to generate flashcards
@app.route("/generate_flashcards", methods=["POST"])
def generate_flashcards():
    try:
        # Parse and clean input
        data = request.get_json()
        input_text = preprocess_text(data.get("text", ""))
        if not input_text:
            return jsonify({"error": "Input text is empty. Please provide valid text."}), 400

        # OpenAI prompt to generate flashcards
        prompt = (
            "Create 10 concise flashcards from the following text. Each flashcard must include:\n"
            "Q: <question>\n"
            "A: <answer>\n\n"
            f"Text: {input_text}"
        )

        # Call OpenAI's API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at creating educational flashcards."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )

        # Extract and parse flashcards from the response
        flashcard_text = response['choices'][0]['message']['content']
        qa_pairs = re.findall(r"Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|\Z)", flashcard_text, re.DOTALL)
        flashcards = [{"question": q.strip(), "answer": a.strip()} for q, a in qa_pairs]

        # Fallback if no flashcards are generated
        if not flashcards:
            flashcards.append({
                "question": "No flashcards generated.",
                "answer": "Try providing more detailed input."
            })

        return jsonify({"flashcards": flashcards})

    except Exception as e:
        print("Error generating flashcards:", e)
        return jsonify({"error": "Failed to generate flashcards. Please try again."}), 500

if __name__ == "__main__":
    app.run(debug=True)
