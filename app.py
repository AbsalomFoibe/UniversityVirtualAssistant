# Import necessary libraries and modules
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import numpy
from flask import Flask, render_template, request
import json
import pickle
import os
import time
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from voc import voc  # Import your custom vocabulary module
import random

# Initialize spaCy for English language processing

nlp = English()
tokenizer = Tokenizer(nlp.vocab)
PAD_Token = 0  # Define a PAD token for padding sequences

# Create a Flask web application
app = Flask(__name__)

# Load your pre-trained neural model
model = models.load_model('mymodel.keras')

# Load your custom dataset and vocabulary using pickle
with open("mydata.pickle", "rb") as f:
    data = pickle.load(f)

# Function to predict a response based on user input
def predict(ques):
    ques = data.getQuestionInNum(ques)
    ques = numpy.array(ques)
    ques = numpy.expand_dims(ques, axis=0)
    y_pred = model.predict(ques)
    res = numpy.argmax(y_pred, axis=1)
    return res

# Function to retrieve a response based on prediction results
def getresponse(results):
    tag = data.index2tags[int(results[0])]
    response = data.response[tag]
    return response

# Main chat function that handles user input and generates responses
def chat(inp):
    inp_x = inp.lower()
    words = inp_x.split()  # Split the input into words

    # Check if any word in the input is not found in the dataset
    for word in words:
        if data.getIndexOfWord(word) == 0:
            return "Sorry, we cannot assist you with that."

    # Predict a response based on user input
    results = predict(inp_x)
    if results == -1:
        return "Sorry, we cannot help you with that."

    # Get a random response based on the prediction results
    response = getresponse(results)
    return random.choice(response)

# Route for the homepage
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle user input and provide responses
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    time.sleep(0.5)  # Add a delay for a more natural conversation flow
    return str(chat(userText))

# Run the Flask application
if __name__ == "__main__":
    app.run()
