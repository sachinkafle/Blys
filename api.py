import joblib
import numpy as np
import spacy
from flask import Flask, request, jsonify
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

# Load pre-trained models and vectorizer
chatbot_model = joblib.load('chatbot_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
knn_model =joblib.load('recommendation_model.pkl')

# like for recommendation, we need customer information
df = pd.read_csv("refined_customer_data.csv")

# Load spaCy for NLP processing
nlp = spacy.load("en_core_web_sm")

# Define Flask app
app = Flask(__name__)

#  function to preprocess text for chatbot like input from the user query
def preprocess(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# Recommendation endpoint
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    customer_id = data.get("customer_id")
    
    # Find the customer data
    customer_data = df[df["Customer_ID"] == customer_id][["Booking_Frequency", "Avg_Spending", "Service_ID"]]
    
    if customer_data.empty:
        return jsonify({"error": "Customer not found"}), 404
    
    
    distances, indices = knn_model.kneighbors(customer_data)
    
    # Get recommended services
    recommendations = df.iloc[indices[0]]["Preferred_Service"].values.tolist()[0]
    return jsonify({"recommendations": recommendations})

# Chatbot endpoint
@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    query = data.get("query", "")
    
    # Preprocess the query
    processed_query = preprocess(query)
    
    # Convert the query into a feature vector
    query_tfidf = vectorizer.transform([processed_query])
    
    # Predict the intent of the query
    intent = chatbot_model.predict(query_tfidf)[0]
    print(intent)
    
    # Respond based on the detected intent
    responses = {
        'reschedule': "Yes, you can reschedule your booking through the Blys app. Would you like me to assist you?",
        'cancel': "Sure, I can help with that. I will proceed with your cancellation request.",
        'price': "The price depends on the service you're interested in. Could you please specify the service?",
        'book': "I can help you with booking. Which service would you like to book?"
    }
    
    response = responses.get(intent, "Sorry, I couldn't understand your request. Could you please rephrase?")
    return jsonify({"response": response})

# Run the Flask app
if __name__ == "__main__":
    app.run(port=5000)
