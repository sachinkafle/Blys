# Blys Assignment


This project provides a comprehensive analysis of customer behavior and implements AI-powered personalization models to enhance customer experience. The solution includes customer segmentation, recommendation systems, and an NLP chatbot for handling customer queries.

## Features

1. **Customer Behavior Analysis**:
   - Synthetic data generation for customer interactions, spending behavior, and sentiment
   - Data preprocessing and normalization
   - Sentiment analysis of customer reviews
   - Customer segmentation using K-Means clustering

2. **AI-Powered Personalization**:
   - Content-based recommendation system using K-Nearest Neighbors
   - Service recommendations based on customer behavior patterns

3. **NLP Chatbot**:
   - Intent classification for customer queries
   - Automated responses for common requests (booking, rescheduling, cancellations, pricing)
   - Integration with the recommendation system

4. **API Development**:
   - Flask-based REST API endpoints for recommendations and chatbot interactions
   - JSON responses for easy integration with frontend applications

## Requirements

To run this project, you'll need the following Python packages:
```
flask==2.2.2
joblib==1.2.0
matplotlib==3.7.1
numpy==1.24.3
pandas==1.5.3
scikit-learn==1.2.2
spacy==3.8.4
textblob==0.19.0
```

## Installation
To install them: 
``` pip install requirements.txt   ```

Since this is only sample data, I have uploaded it in the Github [Keys and Data should be kept private]

Additionally, you'll need to download the spaCy English language model:

``` python -m spacy download en_core_web_sm ```

## Steps for Running the Test

- Step1: Run ```Blys Assignment Sachin Kafle.ipynb``` file for model creation Otherwise download the trained one in the github 
- Step2: Run ``` python api.py``` for running flask app
- Step3: Run ```Blys Calling the API.ipynb``` for inference (output)

## Project Structure

- `Blys Assignment Sachin Kafle.ipynb`: Jupyter notebook containing all analysis and model development
- `refined_customer_data.csv`: Processed customer data
- `recommendation_model.pkl`: Serialized recommendation model
- `chatbot_model.pkl`: Serialized chatbot intent classification model
- `vectorizer.pkl`: Serialized TF-IDF vectorizer for text processing

## How to Use

1. **Customer Analysis**:
   - Run the Jupyter notebook to generate customer segments and insights
   - View visualizations of customer behavior patterns

2. **Recommendation System**:
   - The API endpoint `/recommend` accepts customer IDs and returns personalized service recommendations
   - Example request:
     ```json
     {
       "customer_id": 1001
     }
     ```

3. **Chatbot**:
   - The API endpoint `/chatbot` processes natural language queries and returns appropriate responses
   - Example request:
     ```json
     {
       "query": "Can I reschedule my booking?"
     }
     ```

4. **Running the API**:
   - Execute the Flask application to start the API server
   - The server will run on `http://127.0.0.1:5000`

## Key Insights

- Customer segmentation identifies high-value and at-risk customers
- Sentiment analysis reveals customer satisfaction levels
- The recommendation system suggests services based on similar customer profiles
- The chatbot automates common customer service interactions

---


