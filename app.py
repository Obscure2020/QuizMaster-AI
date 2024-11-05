# This line imports the Flask class and the jsonify function from the Flask library
from flask import Flask, jsonify
#This line creates an instance of the Flask application
app = Flask(__name__)

# This route responds to the root URL (/) and returns a JSON message with the text "Welcome to the Flask backend!"
@app.route('/')

# This function is called when the root URL is accessed, responsible for handling the request and generating a response.
def home():
    return jsonify(message="Welcome to the Flask backend!")

if __name__ == '__main__':
    app.run(debug=True)
