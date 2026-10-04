# SMS Fraud Detection using Machine Learning

## Project Description

This project detects fraudulent and spam SMS messages using Machine Learning and Natural Language Processing (NLP).

The system analyzes the text of an SMS message and classifies it as either SPAM or LEGITIMATE.

The project uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert SMS text into numerical features and Logistic Regression to classify the messages.

A Streamlit web application is also developed to provide an interactive interface where users can enter an SMS message and receive a prediction in real time.

## Algorithms Used

- Logistic Regression
- TF-IDF Vectorization

## Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Plotly

## Dataset

The project uses the SMS Spam Collection dataset, which contains SMS messages labeled as either ham (legitimate) or spam.

The dataset was processed before training the machine learning model.

The preprocessing steps include:

- Selecting the required message and label columns
- Removing unnecessary columns
- Removing duplicate messages
- Cleaning the SMS text
- Converting labels into numerical values
- Splitting the dataset into training and testing data

The cleaned dataset used in the project is:

- `cleaned_spam.csv`

## Project Workflow

1. Load SMS Dataset
2. Data Preprocessing
3. Remove Duplicate Messages
4. Clean SMS Text
5. Convert Labels into Numerical Values
6. Split Dataset into Training and Testing Sets
7. Apply TF-IDF Vectorization
8. Train Logistic Regression Model
9. Evaluate the Model
10. Save the Trained Model and TF-IDF Vectorizer
11. Predict New SMS Messages
12. Display Results using Streamlit

## Machine Learning Model

The project uses Logistic Regression as the classification algorithm.

TF-IDF vectorization converts the text messages into numerical features that can be processed by the machine learning model.

The trained model and vectorizer are saved using Joblib so that the application can make predictions without retraining the model every time it starts.

The saved files are:

- `spam_model.pkl`
- `tfidf_vectorizer.pkl`

## Results

The Logistic Regression model was evaluated on a held-out test dataset.

| Metric | Result |
|---|---:|
| Accuracy | 97.97% |
| Spam Precision | 91% |
| Spam Recall | 95% |
| Spam F1-Score | 93% |

The model achieved approximately 95% recall for spam messages, meaning that it successfully identified most of the spam messages in the test dataset.

## Web Application

The project includes an interactive web application developed using Streamlit.

The application allows users to:

- Enter an SMS message
- Classify the message as SPAM or LEGITIMATE
- View the probability of the prediction


- Compare SPAM and LEGITIMATE probabilities visually
- Receive safety guidance when a message is classified as SPAM

## Project Structure

    SMS-Fraud-Detection-Model/
    │
    ├── assets/
    │   └── download.png
    │
    ├── data/
    │   └── cleaned_spam.csv
    │
    ├── app.py
    ├── preprocess.py
    ├── train_model.py
    ├── test_model.py
    │
    ├── spam_model.pkl
    ├── tfidf_vectorizer.pkl
    │
    ├── requirements.txt
    ├── .gitignore
    └── README.md

## Installation

### 1. Clone the Repository

    git clone https://github.com/komalkadam09/SMS-Fraud-Detection-Model.git

### 2. Open the Project Folder

    cd SMS-Fraud-Detection-Model

### 3. Create a Virtual Environment

    python -m venv venv

### 4. Activate the Virtual Environment

On Windows:

    venv\Scripts\activate

### 5. Install the Required Libraries

    pip install -r requirements.txt

## Run the Application

    streamlit run app.py

The application will open in your web browser.

Enter an SMS message into the application and the trained machine learning model will classify it as SPAM or LEGITIMATE.

## Model Files

The trained machine learning model and TF-IDF vectorizer are stored using Joblib.

- `spam_model.pkl`
- `tfidf_vectorizer.pkl`

## Limitations

This project is a machine learning-based SMS classification system, and its predictions are not guaranteed to identify every spam or fraudulent message.

A message classified as LEGITIMATE should not automatically be considered completely safe.

Users should always be careful when receiving messages containing:

- Unknown links
- Requests for OTPs
- Requests for passwords
- Requests for banking information
- Requests for personal information
- Suspicious payment requests

## Future Improvements

Possible future improvements include:

- Testing additional machine learning algorithms
- Improving text preprocessing
- Expanding the training dataset
- Adding multilingual SMS support
- Adding explainable AI features
- Improving the user interface
- Deploying the application online
- Continuously improving the model using additional labeled data

## Author

Komal Kadam

B.Tech – Artificial Intelligence and Data Science

## Large Files

The trained machine learning model and TF-IDF vectorizer are included in the repository:

- `spam_model.pkl`
- `tfidf_vectorizer.pkl`

The cleaned dataset is also included:

- `cleaned_spam.csv`

The original dataset file `spam.csv` is not included in the GitHub repository because it is excluded using `.gitignore`.

The project can be run using the included cleaned dataset and trained model files.
