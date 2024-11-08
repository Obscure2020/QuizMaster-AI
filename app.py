# Import necessary modules
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    # Initialize variable for submitted text
    submitted_text = ""
    if request.method == 'POST':
        # Get the text input from the form
        submitted_text = request.form.get('student_text', '')
        print("Received input:", submitted_text)  # Debugging log
        # Perform any processing (generating flashcards)
    
    # Render the template with submitted text
    return render_template('index.html', submitted_text=submitted_text)

if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True)
