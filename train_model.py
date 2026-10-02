import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load cleaned dataset
df = pd.read_csv("data/cleaned_spam.csv")


# 2. Separate input and target
X = df["message"]
y = df["label"]


# 3. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 4. Convert SMS text into numerical features using TF-IDF
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


# 5. Create the Naive Bayes model
model = MultinomialNB()

#logistic regression
logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

#train logistic regression model
logistic_model.fit(X_train_tfidf, y_train)

joblib.dump(logistic_model, "spam_model.pkl")

logistic_pred = logistic_model.predict(X_test_tfidf)

print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, logistic_pred))

print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, logistic_pred))

logistic_accuracy = accuracy_score(y_test, logistic_pred)

print("\nLogistic Regression Accuracy:", logistic_accuracy)
print("Logistic Regression Accuracy Percentage:", logistic_accuracy * 100)

# 6. Train the model
model.fit(X_train_tfidf, y_train)


# 7. Make predictions on unseen test data
y_pred = model.predict(X_test_tfidf)


# 8. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("SMS Fraud Detection Model")
print("-------------------------")
print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))
print("TF-IDF features:", X_train_tfidf.shape[1])

print("\nAccuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))