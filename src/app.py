from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import nltk

# Initialize the Flask application
app = Flask(__name__)

# Load the model and vectorizer
with open('svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Download NLTK resources if necessary
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        review_text = request.form['review']

        # Preprocessing
        review_text = review_text.lower()  # Convert to lowercase
        review_text = ''.join(char for char in review_text if char.isalnum() or char.isspace())  # Remove non-word characters
        tokens = nltk.word_tokenize(review_text)  # Tokenization
        stopwords = nltk.corpus.stopwords.words('english')
        tokens = [word for word in tokens if word not in stopwords]  # Remove stopwords

        # Convert tokens back to a single string
        processed_review = ' '.join(tokens)

        # Vectorization
        vectorized_review = vectorizer.transform([processed_review])

        # Prediction
        prediction = model.predict(vectorized_review)

        # Map prediction to sentiment
        sentiment = 'positive' if prediction[0] == 1 else 'negative'

        # Return the prediction
        return render_template('index.html', prediction=sentiment, review=review_text)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
