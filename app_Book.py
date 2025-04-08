import streamlit as st
import pandas as pd
import gdown
import pickle
from tensorflow.keras.models import load_model

# Google Drive file link for 'autoencoder_model_and_data.pkl'
url_rnn_data_and_model = 'https://drive.google.com/uc?export=download&id=1lwNnZvdy0WFm7gl37JxnfFhuGrIQ_GQX'  # Replace with your file ID
output_rnn_data_and_model = 'autoencoder_model_and_data.pkl'

# Download the .pkl file containing both model and reconstructed data
gdown.download(url_rnn_data_and_model, output_rnn_data_and_model, quiet=False)

# Load the model and reconstructed data from the .pkl file
try:
    with open(output_rnn_data_and_model, 'rb') as f:
        loaded_data = pickle.load(f)
        rnn_autoencoder_model = loaded_data['model']
        reconstructed_rnn_long = loaded_data['Book-Rating']
    st.write("Model and reconstructed data loaded successfully.")
except Exception as e:
    st.write(f"Error loading model and data: {e}")

# Function to recommend books for RNN model (without threshold)
def recommend_books_rnn(user_id, num_recommendations=5):
    # Check if the user exists in the dataset
    if user_id in reconstructed_rnn_long['User-ID'].values:
        # Filter ratings for the given user and sort in descending order
        user_ratings = reconstructed_rnn_long[reconstructed_rnn_long['User-ID'] == user_id] \
                        .sort_values(by='Book-Rating', ascending=False)

        # Select top recommended books along with their ratings
        recommended_books_rnn = user_ratings.head(num_recommendations)[['Book-Title', 'Book-Rating']]

        return recommended_books_rnn  # Returning a DataFrame
    else:
        return f"User ID {user_id} not found in the dataset."

# Streamlit user interface
st.header('Book Recommendation System')
st.write('Enter a User ID to get book recommendations.')

user_id_input = st.number_input('Enter User ID:', min_value=1, step=1)

# Button to trigger recommendations
if st.button('Recommend Books'):
    if user_id_input:
        recommended_books = recommend_books_rnn(user_id_input, num_recommendations=5)
        
        if isinstance(recommended_books, pd.DataFrame):
            st.write(f"Recommended books for User {user_id_input}:")
            st.dataframe(recommended_books)
        else:
            st.write(recommended_books)
