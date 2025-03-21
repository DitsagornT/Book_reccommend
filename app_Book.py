
import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Set the correct path to files in Google Drive
drive_path = '/content/drive/MyDrive/project_final/'  # Path to your folder in Google Drive

# Load the model and data from Google Drive
reconstructed_rnn_long = joblib.load(drive_path + "reconstructed_rnn_long.pkl")
rnn_autoencoder_model = load_model(drive_path + "rnn_autoencoder_model.keras")

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
