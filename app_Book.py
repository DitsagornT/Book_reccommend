import streamlit as st
import pandas as pd
import gdown
import pickle
from tensorflow.keras.models import load_model

# Google Drive file link for 'autoencoder_model_and_data.pkl'
url_model = 'https://drive.google.com/uc?export=download&id=1cZzzNkWulSmLTHfgU5Dc1VwcjMEJMowc'  # Replace with your file ID
output_model = 'model_b_5_nan_result.pk'

# Download the .pkl file containing both model and data
gdown.download(url_model, output_model, quiet=False)

with open(output_model, 'rb') as f:
    loaded_data = pickle.load(f)

# ดึงค่าจาก dictionary ที่โหลดมา
autoencoder_b_5 = loaded_data['autoencoder_b_5']
history_b_5_per = loaded_data['history_b_5_per']
merged_df_b_5_per = loaded_data['merged_df_b_5_per']
df_result_filter_missing_b_5_per = loaded_data['df_result_filter_missing_b_5_per']
x_predicted_b_5_per = loaded_data['x_predicted_b_5_per']

st.write("✅ โหลดข้อมูลทั้งหมดเรียบร้อยแล้ว")


# ฟังก์ชันแสดง Top 5 หนังสือที่แนะนำสำหรับ User
def print_top_books_by_user(user_id, df):
    # กรองข้อมูลโดยใช้ User-ID ที่ระบุ
    user_data = df[df['User-ID'] == user_id]
    
    # เรียงข้อมูลตาม Predict-Rating จากมากไปน้อย
    top_books = user_data[['Book-Title', 'Predict-Rating']].sort_values(by='Predict-Rating', ascending=False).head(5)
    
    return top_books

# Streamlit UI
st.header('Book Recommendation System')
st.write('Enter a User ID to get book recommendations.')

user_id_input = st.number_input('Enter User ID:', min_value=1, step=1)

# ปุ่มเพื่อแสดงการแนะนำ
if st.button('Recommend Books'):
    if user_id_input:
        recommended_books = print_top_books_by_user(user_id_input, df_result_filter_missing_b_5_per)
        
        if not recommended_books.empty:
            st.write(f"Recommended books for User {user_id_input}:")
            st.dataframe(recommended_books)
        #else:
            #st.write(f"User-ID not found {user_id_input}")
        else:
            st.write("User-ID not found in the data.")
            st.write("🔍 You can still get recommendations based on a book you like.")
    
            book_list = df_result_filter_missing_b_5_per['Book-Title'].unique()
            selected_book = st.selectbox("Select a book you like:", sorted(book_list))
            rating_input = st.slider("Rate this book (1-10):", min_value=1.0, max_value=10.0, step=0.5)

            if st.button("Recommend Similar Books"):
            # หา user อื่นที่เคยให้คะแนนสูงกับหนังสือที่เลือก
               similar_books_df = df_result_filter_missing_b_5_per[
               (df_result_filter_missing_b_5_per['Book-Title'] != selected_book)
               ]

               # เรียงตาม Predict-Rating สูงสุด
               top_recommendations = similar_books_df.sort_values(by='Predict-Rating', ascending=False).head(5)

               st.write("Recommended books you might also enjoy:")
               st.dataframe(top_recommendations[['Book-Title', 'Predict-Rating']])

