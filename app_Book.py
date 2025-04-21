import streamlit as st
import pandas as pd
import gdown
import pickle
import numpy as np
import tensorflow as tf
import random
import os

# ตั้ง seed เพื่อให้ค่าคงที่
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# Download model (ถ้ายังไม่มี)
url_model = 'https://drive.google.com/uc?export=download&id=1cZzzNkWulSmLTHfgU5Dc1VwcjMEJMowc'
output_model = 'model_b_5_nan_result.pk'
if not os.path.exists(output_model):
    gdown.download(url_model, output_model, quiet=False)

with open(output_model, 'rb') as f:
    loaded_data = pickle.load(f)

# Load data
autoencoder_b_5 = loaded_data['autoencoder_b_5']
merged_df_b_5_per = loaded_data['merged_df_b_5_per']
df_result_filter_missing_b_5_per = loaded_data['df_result_filter_missing_b_5_per']

book_titles = merged_df_b_5_per['Book-Title'].unique()
book_title_to_index = {title: idx for idx, title in enumerate(book_titles)}
input_dim = len(book_titles)

st.title('📚 Book Recommendation System')

# ฟังก์ชันช่วยแนะนำจาก user id
def get_top_books_by_user(user_id):
    df_user = df_result_filter_missing_b_5_per[df_result_filter_missing_b_5_per['User-ID'] == user_id]
    return df_user[['Book-Title', 'Predict-Rating']].sort_values(by='Predict-Rating', ascending=False).head(5)

# ใช้ session_state เก็บสถานะ
if 'mode' not in st.session_state:
    st.session_state.mode = 'user_id'  # default

if 'recommended_books' not in st.session_state:
    st.session_state.recommended_books = None

# ส่วนที่ 1: แนะนำจาก User ID
if st.session_state.mode == 'user_id':
    st.subheader('🔍 Get book recommendations by User ID')
    user_id_input = st.number_input('Enter User ID:', min_value=1, step=1)

    if st.button('Recommend Books'):
        top_books = get_top_books_by_user(user_id_input)
        if not top_books.empty:
            st.session_state.recommended_books = top_books
            st.session_state.mode = 'show_user_result'
        else:
            st.warning(f"User-ID {user_id_input} not found in dataset.")
            st.session_state.mode = 'book_input'

# ส่วนที่ 2: แสดงผลลัพธ์ของ User ID
if st.session_state.mode == 'show_user_result':
    st.success("Recommended books for your User ID:")
    st.dataframe(st.session_state.recommended_books)
    if st.button("Try with a book instead"):
        st.session_state.mode = 'book_input'

# ส่วนที่ 3: แนะนำจากหนังสือที่ชอบ
if st.session_state.mode == 'book_input':
    st.subheader("🎯 Recommend based on a book you like")

    selected_book = st.selectbox("Select a book:", sorted(book_title_to_index.keys()))
    rating_input = st.slider("Rate this book (1-10):", min_value=1.0, max_value=10_
