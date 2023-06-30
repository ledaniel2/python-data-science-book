import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
url = 'https://raw.githubusercontent.com/SK7here/Movie-Review-Sentiment-Analysis/master/IMDB-Dataset.csv'
data = pd.read_csv(url, nrows=250)

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Define a function to preprocess the text
def preprocess_text(text):

    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords and non-alphabetic tokens
    words = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stopwords.words('english')]
    
    # Join the words back into a single string
    preprocessed_text = ' '.join(words)
    
    return preprocessed_text

# Preprocess the movie reviews
data['preprocessed_review'] = data['review'].apply(preprocess_text)

# Initialize the count vectorizer
vectorizer = CountVectorizer(max_features=5000)

# Fit and transform the preprocessed reviews
X = vectorizer.fit_transform(data['preprocessed_review']).toarray()

# Get the sentiment labels
y = data['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Naive Bayes classifier
clf = MultinomialNB()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{cm}')

def analyze_sentiment(text, classifier, vectorizer):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)

    # Convert the preprocessed text to a feature vector
    X_input = vectorizer.transform([preprocessed_text]).toarray()

    # Predict the sentiment using the classifier
    sentiment = classifier.predict(X_input)[0]

    return sentiment

# Example usage of the analyze_sentiment function
new_review = 'A wonderful way to spend time on a too hot summer weekend. This is probably my all-time favorite movie!'
predicted_sentiment = analyze_sentiment(new_review, clf, vectorizer)
print(f'Predicted sentiment: {predicted_sentiment}')
