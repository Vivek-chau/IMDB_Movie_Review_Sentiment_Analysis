# Step 1: Load Libraries and load the model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.models import load_model

# Load the imdb dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key,value in word_index.items()}

# Load the pretrained model with relu activation 
model = load_model('Simple_rnn_imdb.keras')

# Step2 : Helper functions
 # function to decode the reviews
  
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

# Function to preprocess user input

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
     # Check if encoded review is empty or invalid
    if not encoded_review:
        return None
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review



import streamlit as st
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input=st.text_area('Movie Review')

if st.button('classify'):
    preprocessed_input= preprocess_text(user_input)
    # make prediction
    if preprocessed_input is not None:
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0]>0.5 else 'Negative'
        # Display the result
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score:{prediction[0][0]}")
    else:
        st.write('The input review is empty or could not be encoded.')

