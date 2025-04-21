import streamlit as st
import pandas as pd
import gdown
import pickle
import numpy as np
import tensorflow as tf
import random
import os

# ตั้งค่า seed เพื่อให้ผลลัพธ์เหมือนเดิมทุกครั้ง
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# Google Drive file link for 'autoencoder_model_and_data.pkl'
url_model = 'https://drive.google.com/uc?export=download&id=1cZzzNkWulSmLTHfgU5Dc1VwcjMEJMowc'
output_model = 'model_b_5_nan_result.pk'

# Download the .pkl file containing both model and data
if not os.path.exists(output_model):
    gdown.download(url_model, output_model, quiet=False)

with open(output_model, 'rb') as f:
    loaded_data = pickle.load(f)

# ดึงค่าจาก dictionary ที่โหลดมา
autoencoder_b_5 = loaded_data['autoencoder_b_5']
history_b_5_per = loaded_data['history_b_5_per']
merged_df_b_5_per = loaded_data['merged_df_b_5_per']
df_result_filter_missing_b_5_per = loaded_data['df_result_filter_missing_b_5_per']
x_predicted_b_5_per = loaded_data['x_predicted_b_5_per']

st.title('📚 Book Recommendation System')
st.write("โหลดข้อมูลทั้งหมดเรียบร้อยแล้ว")

# สร้าง mapping
book_titles = merged_df_b_5_per['Book-Title'].unique()
book_title_to_index = {title: idx for idx, title in enumerate(book_titles)}
index_to_book_title = {idx: title for title, idx in book_title_to_index.items()}
input_dim = len(book_titles)

# ฟังก์ชันแสดง Top 5 หนังสือที่แนะนำสำหรับ User
def print_top_books_by_user(user_id, df):
    user_data = df[df['User-ID'] == user_id]
    top_books = user_data[['Book-Title', 'Predict-Rating']].sort_values(by='Predict-Rating', ascending=False).head(5)
    return top_books

# UI
st.subheader('🔍 Get book recommendations by User ID')
user_id_input = st.number_input('Enter User ID:', min_value=1, step=1)

if st.button('Recommend Books'):
    recommended_books = print_top_books_by_user(user_id_input, df_result_filter_missing_b_5_per)

    if not recommended_books.empty:
        st.success(f"Top 5 recommended books for User {user_id_input}:")
        st.dataframe(recommended_books)
    else:
        st.warning(f"User-ID {user_id_input} not found in the dataset.")
        st.markdown("### 🎯 Recommend based on a book you like")

        book_list = sorted(book_title_to_index.keys())
        selected_book = st.selectbox("Select a book you like:", book_list)
        rating_input = st.slider("Rate this book (1-10):", min_value=1.0, max_value=10.0, step=0.5)

        if st.button("Recommend Similar Books"):
            user_input_vector = np.full((1, input_dim), -1.0)
            if selected_book in book_title_to_index:
                index = book_title_to_index[selected_book]
                user_input_vector[0, index] = rating_input

                # ใช้ predict แบบ deterministic
                predicted_ratings = autoencoder_b_5.predict(user_input_vector, training=False)

                predicted_df = pd.DataFrame({
                    'Book-Title': book_titles,
                    'Predicted-Rating': predicted_ratings[0]
                })

                predicted_df = predicted_df[predicted_df['Book-Title'] != selected_book]
                top_books = predicted_df.sort_values(by='Predicted-Rating', ascending=False).head(5)

                st.success("Books you may like based on your favorite:")
                st.dataframe(top_books)
            else:
                st.error("Selected book not found in model mapping.")
