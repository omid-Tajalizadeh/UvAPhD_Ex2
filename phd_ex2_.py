


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report

import numpy as np


import nltk
nltk.download('stopwords')
nltk.download('wordnet')

import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import BertTokenizer, BertModel

path = "/data/"

def merge_data_files():
    data_files = [path + 'products-data-0.tsv', path + 'products-data-1.tsv', path + 'products-data-2.tsv', path + 'products-data-3.tsv']
    review_files = [path + 'reviews-0.tsv', path + 'reviews-1.tsv', path + 'reviews-2.tsv', path + 'reviews-3.tsv']

    merged_data = pd.DataFrame()

    for i in range(4):
        data = pd.read_csv(data_files[i], delimiter='\t', header=None, names=['id', 'category', 'product_title'])
        review = pd.read_csv(review_files[i], delimiter='\t', header=None, names=['id', 'rating', 'review_text'])

        # Convert 'id' column to a common data type (e.g., str)
        data['id'] = data['id'].astype(str)
        review['id'] = review['id'].astype(str)

        merged_data = pd.concat([merged_data, pd.merge(data, review, on='id')])

    return merged_data

data = merge_data_files()

data['category'] = data['category'].replace('Ktchen', 'Kitchen')

missing_values = data.isnull().any()

# Check if any missing values exist
if missing_values.any():
    print("Missing values exist in the DataFrame.")
else:
    print("No missing values found in the DataFrame.")


# Check the distribution of the target variable
class_counts = data['category'].value_counts()
print("Class Distribution:\n", class_counts)

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove special characters and digits
    text = re.sub('[^a-zA-Z]', ' ', text)


    return text

def preprocess_data(data):
    # Apply text preprocessing to each text column in the data
    columns = ['product_title','review_text','category']

    for col in columns:
      data[col] = data[col].apply(preprocess_text)

    return data

# Split the data into features and target
pre_data = preprocess_data(data)


X = data[['product_title', 'review_text', 'rating']]
y = data['category']

# Encode the categorical target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TFIDF encoding
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['product_title'] + ' ' + X_train['review_text'])
X_test_tfidf = tfidf_vectorizer.transform(X_test['product_title'] + ' ' + X_test['review_text'])


# transform ratings to scaler values
rating_scaler = StandardScaler()
train_ratings = np.array(X_train["rating"])
X_train_ratings = rating_scaler.fit_transform(train_ratings.reshape(-1,1))
X_train["rating"] = train_ratings

test_ratings = np.array(X_test["rating"])
X_test_ratings = rating_scaler.transform(test_ratings.reshape(-1,1))
X_test["rating"] = test_ratings


# Combine the encoded features
X_train_encoded = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
X_test_encoded = pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Class imbalance handling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train_encoded, y_train)


pipeline = Pipeline([
    ('logreg', LogisticRegression())  # Logistic Regression model
])

# Fit the pipeline on the resampled training data
pipeline.fit(X_resampled, y_resampled)

# Make predictions on the test data
y_pred = pipeline.predict(X_test_encoded)

# Evaluate the model
print(classification_report(y_test, y_pred))