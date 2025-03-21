import streamlit as st
import pandas as pd
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model

# Google Drive file links for 'reconstructed_rnn_long.pkl' and 'rnn_autoencoder_model.keras'
url_rnn_data = 'https://drive.google.com/uc?id=17Mp19o6-dvUNv0DGS5IVR-t-g5xNIlX4'  # Corrected URL for direct download
output_rnn_data = 'reconstructed_rnn_long.pkl'

url_rnn_model = 'https://drive.google.com/uc?id=1--e19kUVXNZ7HnV53Xbj1hXyJtg9_Iak'  # Corrected URL for direct download
output_rnn_model = 'rnn_autoencoder_model.keras'

# Download the model and data
gdown.download(url_rnn_data, output_rnn_data, quiet=False)
gdown.download(url_rnn_model, output_rnn_model, quiet=False)

# Load the reconstructed data (using pandas for .pkl)
try:
    reconstructed_rnn_long = pd.read_pickle(output_rnn_data)
    st.write("Reconstructed data loaded successfully.")
except Exception as e:
    st.write(f"Error loading reconstructed data: {e}")

# Load the RNN Autoencoder model
try:
    rnn_autoencoder_model = load_model(output_rnn_model)
    st.write("RNN Autoencoder model loaded successfully.")
except Exception as e:
    st.write(f"Error loading RNN Autoencoder model: {e}")

# Function to recommend books for RNN model
def recommend_books_rnn(user_id, num_recommendations=5, threshold=5):
    # Check if the user exists in the dataset
    if user_id in reconstructed_rnn_long['User-ID'].values:
        # Filter ratings for the given user and sort in descending order
        user_ratings = reconstructed_rnn_long[reconstructed_rnn_long['User-ID'] == user_id] \
                        .sort_values(by='Book-Rating', ascending=False)

        # Apply threshold to filter recommended books
        user_ratings = user_ratings[user_ratings['Book-Rating'] > threshold]

        # Select top recommended books along with their ratings
        recommended_books_rnn = user_ratings.head(num_recommendations)[['Book-Title', 'Book-Rating']]

        return recommended_books_rnn  # Returning a DataFrame
    else:
        return f"User ID {user_id} not found in the dataset."

# Streamlit user interface
st.title('Book Recommendation System (RNN Autoencoder)')
st.write('Enter a User ID to get book recommendations.')

user_id_input = st.number_input('Enter User ID:', min_value=1, step=1)

# Button to trigger recommendations
if st.button('Recommend Books'):
    if user_id_input:
        recommended_books = recommend_books_rnn(user_id_input, num_recommendations=5, threshold=5)
        
        if isinstance(recommended_books, pd.DataFrame):
            st.write(f"Recommended books for User {user_id_input}:")
            st.dataframe(recommended_books)
        else:
            st.write(recommended_books)
