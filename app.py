import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences



# Load the trained model and tokenizer
model1 = load_model('lstm_model.h5')
model2 = load_model('gru_model.h5')
#lode the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(sequence, verbose=0)
    predicted_word_index = np.argmax(predicted)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app
st.title("Next Word Prediction with LSTM and GRU")
# LSTM input
input_text_lstm = st.text_input(
    "Enter words for LSTM:",
    key="lstm_input"
)
if st.button("Predict Next Word (LSTM)", key="predict_lstm"):
    max_len1 = model1.input_shape[1] + 1
    next_word1 = predict_next_word(model1, tokenizer, input_text_lstm, max_len1)
    st.write(f"LSTM Next Word: {next_word1}")

# GRU input
input_text_gru = st.text_input(
    "Enter words for GRU:",
    key="gru_input"
)
if st.button("Predict Next Word (GRU)", key="predict_gru"):
    max_len2 = model2.input_shape[1] + 1
    next_word2 = predict_next_word(model2, tokenizer, input_text_gru, max_len2)
    st.write(f"GRU Next Word: {next_word2}")


