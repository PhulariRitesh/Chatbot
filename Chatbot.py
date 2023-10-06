"""This code for Chatbot Model is created by : Ritesh Prakashrao Phulari
Teachnook student
as well as Student of IIT Goa
email id="riteshprakashraop@gmail.com
"""
import pickle as pc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler

#Following command was run for downloading module named as keras_tuner
# !pip install keras_tuner

#Following command was run for downloading module named as keras_tuner.engine.hypermodel
# !pip install keras_tuner.engine.hypermodel

# Import necessary libraries
from keras.src.layers import Bidirectional  # Import Bidirectional layer from Keras
from keras_tuner import HyperModel, BayesianOptimization  # Import HyperModel and BayesianOptimization from Keras Tuner
from tensorflow.keras.layers import  Conv1D, GlobalMaxPooling1D  # Import various layers from TensorFlow Keras
from tensorflow.keras.preprocessing.text import Tokenizer  # Import Tokenizer for text preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Import pad_sequences for padding sequences
from tensorflow.keras.optimizers import Adam  # Import Adam optimizer
from tensorflow.keras.losses import SparseCategoricalCrossentropy  # Import SparseCategoricalCrossentropy loss
from sklearn.utils.class_weight import compute_class_weight  # Import compute_class_weight for calculating class weights
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # Import callbacks for model training
from kerastuner.tuners import RandomSearch  # Import RandomSearch tuner from Keras Tuner
from tensorflow.keras.losses import SparseCategoricalCrossentropy  # Import SparseCategoricalCrossentropy loss from TensorFlow Keras
from nltk.corpus import stopwords  # Import stopwords from NLTK corpus
import re  # Import re module for regular expressions
import nltk  # Import NLTK library for natural language processing

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, roc_auc_score

"""<FONT COLOR="BLUE">*DATA PREPROCESSING BY REMOVING SPECIAL CHARACTERS*</FONT>"""

import nltk
import re

nltk.download('stopwords')
nltk.download('punkt')

from nltk.tokenize import word_tokenize

def preprocess_text(text):
    # Remove special characters and numbers using regular expressions
    text = re.sub(r'[^A-Za-z\s]', '', text)

    # Convert the text to lowercase
    text = text.lower()

    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    tokens = word_tokenize(text)

    return tokens

"""<font color="yellow">**Feature Extraction Using TF-IDF (Term Frequency-Inverse Document Frequency)**</font>"""

from sklearn.feature_extraction.text import TfidfVectorizer


# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

"""<FONT COLOR="Orange">**Data READING** and *Seperating data in different forms test data,training data,validation data*</FONT>"""

training_set_1 = pd.read_csv('/content/drive/MyDrive/AI /Train.csv')
test_set_1 = pd.read_csv('/content/drive/MyDrive/AI /Test.csv')
valid_set_1 = pd.read_csv('/content/drive/MyDrive/AI /Valid.csv')

data = [
    ("I love this product!", "positive"),
    ("Terrible experience, never buying again.", "negative"),
    ("It's okay, nothing special.", "neutral"),
    ("This movie was fantastic!", "positive"),
    ("Worst restaurant ever.", "negative"),
    ("Average performance.", "neutral"),
    ("The weather is so beautiful today!", "positive"),
    ("I'm so disappointed with the service.", "negative"),
    ("No strong feelings either way.", "neutral"),
    ("This book is amazing.", "positive"),
    ("I regret buying this.", "negative"),
    ("I don't really have an opinion.", "neutral"),
    ("Absolutely wonderful experience!", "positive"),
    ("I can't stand this anymore.", "negative"),
    ("Not much to say about it.", "neutral"),
    ("The concert was incredible!", "positive"),
    ("Awful customer support.", "negative"),
    ("I'm not sure how I feel about it.", "neutral"),
    ("Delicious food and great ambiance.", "positive"),
    ("I'm furious about this situation.", "negative"),
    ("It's just average, nothing remarkable.", "neutral"),
    ("Loved every minute of it!", "positive"),
    ("Couldn't be more dissatisfied.", "negative"),
    ("It's alright, I guess.", "neutral"),
    ("The product exceeded my expectations!", "positive"),
    ("Horrible quality, waste of money.", "negative"),
    ("I have mixed feelings about this.", "neutral"),
    ("Best service I've ever received.", "positive"),
    ("I'm really angry with the company.", "negative"),
    ("Not bad, but not great either.", "neutral"),
    ("Incredibly enjoyable experience!", "positive"),
    ("This is a complete disaster.", "negative"),
    ("It's decent, nothing too special.", "neutral"),
    ("I'm thrilled with the results!", "positive"),
    ("I'm utterly disgusted.", "negative"),
    ("I'm indifferent to this.", "neutral")
    ("What's wrong with you.","negative")
]

corpus = [preprocess_text(text) for text, _ in data]

training_set_1.head()

# Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(training_set_1)

test_set_1.head()
training_set_1.head()

pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)

print(training_set_1.shape)
print(test_set_1.shape)

training_set_1.info()

plt.style.use('fivethirtyeight')
sns.pairplot(training_set_1)

"""# <FONT COLOR="ORANGE">DATA PREPROCESSING:</FONT>"""

# Tokenization and padding
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')  # Initialize Tokenizer with a vocabulary size limit
tokenizer.fit_on_texts(corpus)  # Fit Tokenizer on preprocessed text data
word_index = tokenizer.word_index  # Get word index from Tokenizer
sequences = tokenizer.texts_to_sequences(corpus)  # Convert text to sequences of indices
padded_sequences = pad_sequences(sequences, padding='post')

training_set_1.isnull().sum()

test_set_1.isnull().sum()

scale = MaxAbsScaler()
traning_set = scale.fit_transform(tfidf_matrix)
training_set_1

x_train = training_set_1[0:40000]
y_train = training_set_1[1:40002]

X_train = tfidf_matrix[2:40000]
Y_train= tfidf_matrix[1:40001]

# Convert sentiment labels to numerical values
sentiment_mapping = {"positive": 0, "negative": 1, "neutral": 2}  # Mapping of sentiment labels to numerical values
labels = np.array([sentiment_mapping[sentiment] for _, sentiment in data])  # Convert sentiment labels to numerical values

training_set = training_set_1.iloc[:,1:2].values
test_set = test_set_1.iloc[:, 1:2].values
training_set

print(f"Befor Reshape {x_train.shape}")
x_train = np.reshape(x_train,(40000,2))

# Split the data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)  # Split data into training and validation sets

"""## <font color="gold">**MODEL BUILDING:**</font>"""

x_train = training_set_1[0:39999]
y_train = training_set_1[1:40002]

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Calculate class weights for imbalanced dataset
class_counts = np.bincount(train_labels)  # Count the occurrences of each class in the training labels
total_samples = sum(class_counts)  # Calculate the total number of samples
class_weights = {cls: total_samples / count for cls, count in enumerate(class_counts)}  # Calculate class weights

Sentiment = Sequential()

Sentiment.add(LSTM(units = 4,activation = 'sigmoid', input_shape = (None,1)))

Sentiment.add(Dense(units = 1))

Sentiment.compile(optimizer = 'adam',loss = 'mean_squared_error')

class SentimentHyperModel(HyperModel):
    def build(self, hp):
        # Build the model architecture with hyperparameters
        chatbot = Sequential()  # Initialize a Sequential model
        chatbot.add(Embedding(len(word_index) + 1, hp.Int('embedding_dim', min_value=64, max_value=256, step=32)))  # Add an Embedding layer
        chatbot.add(Bidirectional(LSTM(hp.Int('lstm_units', min_value=64, max_value=128, step=32), return_sequences=True)))  # Add a Bidirectional LSTM layer
        chatbot.add(GlobalMaxPooling1D())  # Add a GlobalMaxPooling1D layer
        chatbot.add(Dense(hp.Int('dense_units', min_value=64, max_value=256, step=32), activation='relu'))  # Add a Dense layer with ReLU activation
        chatbot.add(Dropout(hp.Float('dense_dropout', min_value=0.2, max_value=0.5, step=0.1)))  # Add a Dropout layer
        chatbot.add(Dense(3, activation='softmax'))  # Add a Dense layer with softmax activation
        optimizer = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1, sampling='LOG'))  # Initialize Adam optimizer with a learning rate
        loss = SparseCategoricalCrossentropy()  # Use SparseCategoricalCrossentropy loss
        chatbot.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])  # Compile the model
        return chatbot

from sklearn.model_selection import train_test_split

# Assume X contains your features and y contains your labels
X_train, X_temp, y_train, y_temp = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Print the shapes of the sets to verify the split
print("Train set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)
print("Test set:", X_test.shape, y_test.shape)

# Instantiate the custom hypermodel
hypermodel = SentimentHyperModel()

from tensorflow.keras.layers import Embedding
import tensorflow as tf

X_test = tfidf_matrix[2:4000]
y_test = tfidf_matrix[2:4000]
# Compile the model
Sentiment.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

X_test_shape = X_test.shape
X_test_new = tf.convert_to_tensor(X_test_shape)

y_test_shape = y_test.shape
y_test_new = tf.convert_to_tensor(y_test_shape)

# Define the indices of the array you want to remove
start_index = 0  # Starting index of the array to remove along the first dimension
end_index = 2    # Ending index (exclusive) of the array to remove along the first dimension

# Remove the array by slicing
X_train_sliced = X_train[start_index:end_index]
y_train_sliced = y_train[start_index:end_index]

"""<font color="GOLD" size="6px">**TRAINING THE MODEL :**<FONT>"""

# Use Bayesian optimization for hyperparameter tuning
tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=100,
    directory='my_dir',
    project_name='sentiment_analysis'
)

# Perform the hyperparameter search using training data
tuner.search(train_texts, train_labels, epochs=100, validation_data=(val_texts, val_labels),
             class_weight=class_weights, callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=2)
    ])

"""# Handling overfitting and underfitting :"""

# Load and use the best hyperparameters found by the tuner
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model using the best hyperparameters
best_model = hypermodel.build(best_hps)
print(best_model)

"""# <FONT COLOR="ORANGE"> SENTIMENT ANALYSIS :</FONT>"""

# Function to predict sentiment using the trained best model
def predict_sentiment(user_input, model):
    preprocessed_input = preprocess_text(user_input)  # Preprocess user input
    sequence = tokenizer.texts_to_sequences([preprocessed_input])  # Convert preprocessed input to sequences
    padded_sequence = pad_sequences(sequence, padding='post', maxlen=100)  # Pad the sequence
    sentiment_probabilities = model.predict(padded_sequence)[0]  # Predict sentiment probabilities
    predicted_sentiment = np.argmax(sentiment_probabilities)  # Get the index of the predicted sentiment
    return list(sentiment_mapping.keys())[list(sentiment_mapping.values()).index(predicted_sentiment)]  # Map index to sentiment label

# Load the best model after tuning
best_model = tuner.get_best_models(num_models=1)[0]

# User interaction loop for sentiment prediction
while True:
    user_input = input("Enter your message (or 'exit' to quit): ")  # Get user input
    if user_input.lower() == 'exit':  # Check if user wants to exit
        break
    sentiment = predict_sentiment(user_input, best_model)  # Predict sentiment
    print("Predicted sentiment:", sentiment)  # Print predicted sentiment

"""# Prediction :

## For test_data :
"""

from sklearn.preprocessing import MinMaxScaler
real_sentiment = test_set
inputs = real_sentiment

inputs = np.reshape(inputs, (5000, 1))           # reshape the data
predicted_Sentiment = Sentiment.predict(inputs)     # prediction

"""## For training data :"""

training_set = pd.read_csv('/content/drive/MyDrive/AI /Train.csv')
training_set = training_set.iloc[:, 1:2].values

real_Sentiment = training_set
inputs = real_Sentiment

inputs = np.reshape(inputs, (40000,1))                # reshape the data
predicted_stock_price = Sentiment.predict(inputs)     # prediction




def chat_with_bot():
    print("Welcome to the Sentiment Analysis Chatbot!")
    print("You can type 'exit' to quit the chat.")

    while True:
        user_input = input("You: ")  # Get user input
        if user_input.lower() == 'exit':  # Check if user wants to exit
            print("Goodbye!")
            break
        sentiment = predict_sentiment(user_input, best_model, tokenizer)  # Predict sentiment
        print("Chatbot: Predicted sentiment -", sentiment)  # Print predicted sentiment

# Start the chatbot
chat_with_bot()

#Following command was run for downloading module named as flask
# pip install flask

pc.dump(chat_bot,open('model.pk1','wb'))
Chat = pickle.load(open('model.pk1','rb'))

