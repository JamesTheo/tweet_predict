import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

model_path = "tweet_model.h5"
model = load_model(model_path)

# Function to process the user input
def process_text(input_text):
    # Your NLP model processing code goes here
    # Replace the print statement with your model's inference code
    tokenizer = Tokenizer(num_words=10000)  # Consider top 10,000 words
    max_sequence_length=100
    my_tweet = [input_text]
    new_sequences_1 = tokenizer.texts_to_sequences(my_tweet)
    new_data_1 = pad_sequences(new_sequences_1, maxlen=max_sequence_length)
    new_predictions_1 = np.round(model.predict(new_data_1)).flatten()
    label = "Real Disaster" if new_predictions_1[0] == 1 else "Not about Real Disaster"
    
    return label

# Streamlit app layout
st.title("NLP Model Demo")
st.write("Enter some text below:")

# Text input box
user_input = st.text_area("Input Text", "")

# Process button
if st.button("Process"):
    if user_input:
        processed_result = process_text(user_input)
        st.write("Processed Result:")
        st.write(processed_result)
    else:
        st.write("Please enter some text.")
