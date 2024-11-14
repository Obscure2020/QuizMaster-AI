import openai
import os
import re
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Route to serve the HTML page
@app.route("/")
def home():
    return render_template("index.html")

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^\w\s.,?!]', '', text)  # Keep only words, spaces, punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()

# Route to handle flashcard generation
@app.route("/generate_flashcards", methods=["POST"])
def generate_flashcards():
    data = request.get_json()
    input_text = preprocess_text(data.get("text", ""))

    prompt = "Generate ten flashcards with short 3-4 word answers in the format: 'Q: <question>' and 'A: <answer>'."


    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text}
            ],
            max_tokens=300,
            temperature=0.7
        )

        flashcard_text = response['choices'][0]['message']['content']
        print("Raw model output:", flashcard_text)

        # Initialize list to hold parsed flashcards
        flashcards = []

        # Use regex to find each Q: and corresponding A:
        qa_pairs = re.findall(r"Q:\s*(.*?)\nA:\s*(.*?)(?=\nQ:|\Z)", flashcard_text, re.DOTALL)
        for question, answer in qa_pairs:
            flashcards.append({"question": question.strip(), "answer": answer.strip()})

        # Fallback message if no flashcards are generated
        if not flashcards:
            flashcards.append({
                "question": "No structured flashcards were generated.",
                "answer": "The input text may not have had enough detail or clarity."
            })

        print("Parsed flashcards:", flashcards)

    except Exception as e:
        print("Error encountered:", e)
        return jsonify({"error": str(e)}), 500

    return jsonify({"flashcards": flashcards})

if __name__ == "__main__":
    app.run(debug=False)
