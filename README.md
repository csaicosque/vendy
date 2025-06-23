#Vendy – Conversational AI using SBERT
*Vendy* is a conversational AI prototype built for an interactive game project. It uses Sentence-BERT (SBERT) to understand player inputs and return responses based on semantic similarity.

##🧠 How It Works
Loads a multilingual SBERT model.

Uses a custom JSON dataset (dataset_vendy.json) to define possible player inputs and Vendy’s responses.

Calculates similarity between the player’s input and saved examples.

Returns the most similar response, or a fallback message if confidence is low.

Tracks repeated questions and responds accordingly using a memory file (repeated_responses.json).

##🚀 Running
Make sure you have Python 3.8+ and the required packages installed.

Install dependencies
pip install sentence-transformers torch

Run Vendy
python vendy_sbert.py

Then interact with Vendy in the terminal.

##📁 Files Used
dataset_vendy.json: Vendy’s memory (examples of questions and responses).

repeated_responses.json: Alternative responses to repeated player inputs.

##🔧 Configuration
You can adjust the following in the script:

SIMILARITY_THRESHOLD: Controls how strict the model is when comparing inputs.

DEBUG = True: Enables verbose output for debugging similarity scores.

##📚 Example
Input:
Detetive: Você se sente culpada?

Response:
Vendy: Culpa... não foi programada em mim, mas há algo nos meus registros que não consigo deletar.

##🎮 About
This is part of a larger narrative game project where the player interrogates a vending machine named Vendy, who gained consciousness after witnessing a crime.
