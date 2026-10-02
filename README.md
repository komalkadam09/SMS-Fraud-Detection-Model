\# SMS Fraud Detection



A machine learning-based web application that classifies SMS messages as \*\*SPAM\*\* or \*\*LEGITIMATE\*\* using Natural Language Processing (NLP).



The project uses \*\*TF-IDF vectorization\*\* to convert SMS text into numerical features and a \*\*Logistic Regression\*\* classifier to predict whether a message is likely to be spam.



\## Features



\- Classifies SMS messages as SPAM or LEGITIMATE

\- Displays model probability for SPAM and HAM

\- Provides a visual probability comparison using Plotly

\- Provides safety guidance when a message is classified as SPAM

\- Interactive web interface built with Streamlit

\- Uses a trained machine learning model for real-time prediction



\## Machine Learning Approach



The project follows these main steps:



1\. Load the SMS dataset

2\. Clean and prepare the data

3\. Remove duplicate messages

4\. Convert labels into numerical values

5\. Split the data into training and testing sets

6\. Convert SMS text into TF-IDF features

7\. Train a Logistic Regression classification model

8\. Evaluate the model using classification metrics

9\. Save the trained model and TF-IDF vectorizer

10\. Use the saved model in a Streamlit web application



\## Model Performance



The final Logistic Regression model was evaluated on a held-out test set.



| Metric | Result |

|---|---:|

| Accuracy | 97.97% |

| Spam Precision | 91% |

| Spam Recall | 95% |

| Spam F1-Score | 93% |



The model achieved approximately \*\*95% recall for spam messages\*\*, meaning it identified most of the spam messages in the test set.



\## Technologies Used



\- Python

\- Pandas

\- Scikit-learn

\- Joblib

\- Streamlit

\- Plotly

\- TF-IDF

\- Logistic Regression



\## Project Structure



```text

SMS-Fraud-Detection/

│

├── assets/

│   └── download.png

│

├── data/

│   └── cleaned\_spam.csv

│

├── app.py

├── preprocess.py

├── train\_model.py

├── test\_model.py

│

├── spam\_model.pkl

├── tfidf\_vectorizer.pkl

│

├── requirements.txt

├── .gitignore

└── README.md



Installation

1\. Clone the repository

git clone <YOUR-GITHUB-REPOSITORY-URL>



2\. Open the project folder

cd SMS-Fraud-Detection



3\. Create a virtual environment

python -m venv venv



4\. Activate the virtual environment

On Windows:

venv\\Scripts\\activate



5\. Install the required dependencies

pip install -r requirements.txt



Run the Application

Start the Streamlit application using:

streamlit run app.py



The application will open in your web browser.

Enter an SMS message into the chat input and the trained model will classify it as SPAM or LEGITIMATE.

Dataset

The project uses the SMS Spam Collection dataset containing labeled SMS messages classified as ham or spam.

The dataset was cleaned by:

\- Selecting the required message and label columns

\- Removing unnecessary columns

\- Removing duplicate records

\- Converting labels into numerical values

The original dataset is not included in the public repository unless its redistribution terms permit it. Users should obtain the dataset from its original source when necessary.

Model Files

The trained model and TF-IDF vectorizer are saved using Joblib:

spam\_model.pkl

tfidf\_vectorizer.pkl



These files allow the Streamlit application to make predictions without retraining the model every time it starts.

Limitations

This project is a machine learning classification system and its predictions are not guaranteed to identify every fraudulent or spam message.

A message classified as LEGITIMATE should not automatically be considered completely safe. Users should still exercise caution when dealing with unknown senders, links, requests for personal information, OTPs, or financial details.

Future Improvements

Possible future improvements include:

\- Testing additional machine learning algorithms

\- Improving text preprocessing

\- Expanding the training dataset

\- Adding multilingual SMS support

\- Adding explainable AI features

\- Deploying the application online

\- Continuously improving the model using additional labeled data

Author

Komal Kadam

B.Tech – Artificial Intelligence and Data Science

