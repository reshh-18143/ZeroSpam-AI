# Text Classification Project

This project is focused on text classification using various machine learning techniques and libraries. The goal is to preprocess text data and classify it using a machine learning model.

## Dependencies

To run this project, you need to have the following libraries installed:

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pickle
from win32com.client import Dispatch
import tkinter as tk


You can install the required libraries using:
pip install pandas numpy scikit-learn nltk pywin32

Dataset
The dataset should be loaded into a pandas DataFrame. Make sure to preprocess the text data by removing stopwords, lemmatizing, and applying TF-IDF vectorization.

Usage
Preprocess Data: Preprocess the text data by cleaning it, removing stopwords, and lemmatizing.

TF-IDF Vectorization: Convert the text data into TF-IDF vectors:

python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
Train-Test Split: Split the data into training and testing sets:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Train Model: Train a machine learning model such as Multinomial Naive Bayes:

model = MultinomialNB()
model.fit(X_train, y_train)
Evaluate Model: Evaluate the performance of the model on the test set:

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

Save Model: Save the trained model and vectorizer using pickle:
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)



Acknowledgments
This project is built using open-source libraries such as pandas, numpy, scikit-learn, and nltk.

Special thanks to the contributors and the open-source community for their valuable resources and support.
