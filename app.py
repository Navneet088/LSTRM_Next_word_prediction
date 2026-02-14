from xml.parsers.expat import model
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
st.title("LSTM Next Word Prediction")
input_text = st.text_input("Enter a sequence of words:")
if st.button("Predict Next Word"):  
    max_sequence_len = model1.input_shape[1] + 1
    next_word = predict_next_word(model1, tokenizer, input_text, max_sequence_len)
    st.write(f"Predicted Next Word: {next_word}")


st.title("GRU Next Word Prediction")
input_text = st.text_input("Enter a sequence of words:")    
if st.button("Predict Next Word GRU"):  
    max_sequence_len = model2.input_shape[1] + 1
    next_word = predict_next_word(model2, tokenizer, input_text, max_sequence_len)
    st.write(f"Predicted Next Word: {next_word}")


