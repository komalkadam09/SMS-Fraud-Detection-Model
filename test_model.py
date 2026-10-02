import joblib

# Load saved model and vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("spam_model.pkl")

# Test SMS
message = ["Hey, are we meeting tomorrow at 10 am?"]

# Convert message to TF-IDF
message_tfidf = vectorizer.transform(message)

# Predict
prediction = model.predict(message_tfidf)

if prediction[0] == 1:
    print("Prediction: SPAM")
else:
    print("Prediction: HAM")