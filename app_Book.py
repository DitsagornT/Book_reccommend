import streamlit as st
import pandas as pd
import gdown
import pickle
from tensorflow.keras.models import load_model
import numpy as np 
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

st.write("โหลดข้อมูลทั้งหมดเรียบร้อยแล้ว")
# สร้าง mapping ถ้ายังไม่ได้ทำ
book_titles = merged_df_b_5_per['Book-Title'].unique()
book_title_to_index = {title: idx for idx, title in enumerate(book_titles)}
index_to_book_title = {idx: title for title, idx in book_title_to_index.items()}
input_dim = len(book_titles)  # จำนวนหนังสือทั้งหมด


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
            st.write("You can still get recommendations based on a book you like.")

            book_list = sorted(book_title_to_index.keys())
            selected_book = st.selectbox("Select a book you like:", book_list)
            rating_input = st.slider("Rate this book (1-10):", min_value=1.0, max_value=10.0, step=0.5)

           if st.button("Recommend Similar Books"):
              # ✅ สร้าง input vector
              user_input_vector = np.full((1, input_dim), -1.0)  # ใช้ค่า -1 แทน missing

              # ใส่คะแนนที่ผู้ใช้เลือก
              if selected_book in book_title_to_index:
                  index = book_title_to_index[selected_book]
                  user_input_vector[0, index] = rating_input

                  # ✅ ทำนายด้วย autoencoder
                  predicted_ratings = autoencoder_b_5.predict(user_input_vector)

                  # สร้าง DataFrame แสดงผล
                  predicted_df = pd.DataFrame({
                            'Book-Title': book_titles,
                            'Predicted-Rating': predicted_ratings[0]
                  })

                  # ลบหนังสือที่ผู้ใช้เลือกออก
                  predicted_df = predicted_df[predicted_df['Book-Title'] != selected_book]

                  # แสดงผลลัพธ์
                  top_books = predicted_df.sort_values(by='Predicted-Rating', ascending=False).head(5)
                  st.write("📚 Recommended books based on your favorite:")
                  st.dataframe(top_books)
              else:
                  st.warning("❌ Book not found in model mapping.")

