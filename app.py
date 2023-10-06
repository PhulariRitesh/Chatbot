from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ... (Import other necessary modules and functions)
app = Flask(__name__)

Chat = pickle.load(open('model.pk1','rb'))

@app.route('/predict', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        sentiment = predict_sentiment(user_input)
        return jsonify({'message': 'Chatbot: Predicted sentiment - ' + sentiment})

    return render_template('index.html')

def predict_sentiment(user_input):
    # Preprocess user input
    preprocessed_input = preprocess_text(user_input)
    sequence = tokenizer.texts_to_sequences([preprocessed_input])
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=100)

    # Predict sentiment
    sentiment_probabilities = best_model.predict(padded_sequence)[0]
    predicted_sentiment = np.argmax(sentiment_probabilities)

    return list(sentiment_mapping.keys())[list(sentiment_mapping.values()).index(predicted_sentiment)]

if __name__ == '__main__':
    app.run(debug=True)
