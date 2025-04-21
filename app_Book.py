import streamlit as st
import pandas as pd
import gdown
import pickle
import numpy as np
import tensorflow as tf
import random
import os

# ตั้งค่า seed
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# โหลด model
url_model = 'https://drive.google.com/uc?export=download&id=1cZzzNkWulSmLTHfgU5Dc1VwcjMEJMowc'
output_model = 'model_b_5_nan_result.pk'

if not os.path.exists(output_model):
    gdown.download(url_model, output_model, quiet=False)

with open(output_model, 'rb') as f:
    loaded_data = pickle.load(f)

# Extract data
autoencoder_b_5 = loaded_data['autoencoder_b_5']
merged_df_b_5_per = loaded_data['merged_df_b_5_per']
df_result_filter_missing_b_5_per = loaded_data['df_result_filter_missing_b_5_per']

book_titles = merged_df_b_5_per['Book-Title'].unique()
book_title_to_index = {title: idx for idx, title in enumerate(book_titles)}
input_dim = len(book_titles)

st.title('📚 Book Recommendation System')

# จัดการสถานะด้วย session_state
if 'mode' not in st.session_state:
    st.session_state.mode = 'user_id'

if 'user_result' not in st.session_state:
    st.session_state.user_result = None

if 'book_result' not in st.session_state:
    st.session_state.book_result = None

# ฟังก์ชันแนะนำจาก user id
def get_top_books_by_user(user_id):
    user_data = df_result_filter_missing_b_5_per[df_result_filter_missing_b_5_per['User-ID'] == user_id]
    return user_data[['Book-Title', 'Predict-Rating']].sort_values(by='Predict-Rating', ascending=False).head(5)

# โหมดแนะนำจาก user id
if st.session_state.mode == 'user_id':
    st.subheader('🔍 Get book recommendations by User ID')
    user_id_input = st.number_input('Enter User ID:', min_value=1, step=1)

    if st.button('Recommend Books'):
        top_books = get_top_books_by_user(user_id_input)
        if not top_books.empty:
            st.session_state.user_result = top_books
            st.session_state.mode = 'user_result'
        else:
            st.warning(f"User-ID {user_id_input} not found. Try recommending from your favorite book.")
            st.session_state.mode = 'book_input'

# แสดงผลลัพธ์จาก user id
if st.session_state.mode == 'user_result':
    st.subheader("📖 Recommendations for Your User ID")
    st.dataframe(st.session_state.user_result)
    if st.button("Try recommending from a favorite book instead"):
        st.session_state.mode = 'book_input'

# แนะนำจากหนังสือที่ชอบ
if st.session_state.mode == 'book_input':
    st.subheader("🎯 Recommend based on a book you like")

    book_list = sorted(book_title_to_index.keys())
    selected_book = st.selectbox("Select a book you like:", book_list)
    rating_input = st.slider("Rate this book (1-10):", min_value=1.0, max_value=10.0, step=0.5)

    if st.button("Recommend Similar Books"):
        user_input_vector = np.full((1, input_dim), -1.0)
        user_input_vector[0, book_title_to_index[selected_book]] = rating_input
        predicted_ratings = autoencoder_b_5.predict(user_input_vector, training=False)

        predicted_df = pd.DataFrame({
            'Book-Title': book_titles,
            'Predicted-Rating': predicted_ratings[0]
        })

        predicted_df = predicted_df[predicted_df['Book-Title'] != selected_book]
        top_books = predicted_df.sort_values(by='Predicted-Rating', ascending=False).head(5)

        st.session_state.book_result = top_books
        st.session_state.mode = 'book_result'

# แสดงผลลัพธ์จากหนังสือที่ชอบ
if st.session_state.mode == 'book_result':
    st.subheader("📚 Books similar to your favorite")
    st.dataframe(st.session_state.book_result)
    if st.button("Back to User ID Input"):
        st.session_state.mode = 'user_id'
