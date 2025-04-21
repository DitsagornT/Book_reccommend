import streamlit as st
import pandas as pd
import gdown
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Google Drive file link
url_model = 'https://drive.google.com/uc?export=download&id=1cZzzNkWulSmLTHfgU5Dc1VwcjMEJMowc'
output_model = 'model_b_5_nan_result.pk'

# Download the .pkl file
gdown.download(url_model, output_model, quiet=False)

# Load the model and data
with open(output_model, 'rb') as f:
    loaded_data = pickle.load(f)

autoencoder_b_5 = loaded_data['autoencoder_b_5']
history_b_5_per = loaded_data['history_b_5_per']
merged_df_b_5_per = loaded_data['merged_df_b_5_per']
df_result_filter_missing_b_5_per = loaded_data['df_result_filter_missing_b_5_per']
x_predicted_b_5_per = loaded_data['x_predicted_b_5_per']

st.write("✅ โหลดข้อมูลเรียบร้อยแล้ว")

# สร้าง mapping
book_titles = merged_df_b_5_per['Book-Title'].unique()
book_title_to_index = {title: idx for idx, title in enumerate(book_titles)}
input_dim = len(book_titles)

# ฟังก์ชันแนะนำหนังสือจาก user ID
def print_top_books_by_user(user_id, df):
    user_data = df[df['User-ID'] == user_id]
    top_books = user_data[['Book-Title', 'Predict-Rating']].sort_values(by='Predict-Rating', ascending=False).head(5)
    return top_books

# Header
st.header('📚 Book Recommendation System')

# รักษาสถานะ
if 'show_book_select' not in st.session_state:
    st.session_state['show_book_select'] = False

user_id_input = st.number_input('🔢 Enter User ID:', min_value=1, step=1)

if st.button('🎯 Recommend Books'):
    if user_id_input in df_result_filter_missing_b_5_per['User-ID'].values:
        recommended_books = print_top_books_by_user(user_id_input, df_result_filter_missing_b_5_per)
        st.write(f"📖 Recommended books for User {user_id_input}:")
        st.dataframe(recommended_books)
        st.session_state['show_book_select'] = False  # ปิดส่วน book select ถ้ามี user ID แล้ว
    else:
        st.warning("❌ User ID not found in the dataset.")
        st.session_state['show_book_select'] = True  # แสดงตัวเลือกจากหนังสือที่ชอบ

# หากไม่พบ user id และต้องแนะนำจากหนังสือที่ชอบ
if st.session_state['show_book_select']:
    st.write("🔍 ลองเลือกหนังสือที่คุณชอบ เพื่อให้โมเดลแนะนำเล่มอื่นให้")

    book_list = sorted(book_title_to_index.keys())
    selected_book = st.selectbox("👍 Select a book you like:", book_list)
    rating_input = st.slider("⭐ Rate this book (1-10):", min_value=1.0, max_value=10.0, step=0.5)

    if st.button("✨ Recommend Similar Books"):
        user_input_vector = np.full((1, input_dim), -1.0)
        index = book_title_to_index.get(selected_book, None)
        if index is not None:
            user_input_vector[0, index] = rating_input
            predicted_ratings = autoencoder_b_5.predict(user_input_vector)
            predicted_df = pd.DataFrame({
                'Book-Title': book_titles,
                'Predicted-Rating': predicted_ratings[0]
            })
            predicted_df = predicted_df[predicted_df['Book-Title'] != selected_book]
            top_books = predicted_df.sort_values(by='Predicted-Rating', ascending=False).head(5)
            st.write("📚 Recommended books based on your favorite:")
            st.dataframe(top_books)
        else:
            st.warning("❌ Book not found in model mapping.")
