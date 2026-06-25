# Import library yang dibutuhkan
import tensorflow as tf
from tensorflow import keras
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time


# st.title("Prediksi Kualitas Tidur dan Tingkat Stres")
st.markdown("<h2 style='text-align: center;'>Prediksi Kualitas Tidur dan Tingkat Stres</h2>", unsafe_allow_html=True)
st.info("Info : 1 Gelas Kopi Kurang Lebih 95 mg Kafein")


col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Jenis Kelamin", ["Pria", "Perempuan", "Lainnya"])
    age = st.slider("Umur (Tahun)", 5, 100, 24)
    tinggi = st.slider("Tinggi (cm)", 50.0, 250.0, 165.0)
    berat = st.slider("Berat (kg)", 10.0, 300.0, 70.0)
    smoking = st.selectbox("Apakah Anda Merokok ?", ["Tidak", "Ya"])
with col2:
    coffee = st.slider("Kopi (Gelas)", 0.0, 10.0, 1.0)
    caffeine = st.slider("Kafein (mg)", 0.0, 1000.0, 95.0)
    sleep = st.slider("Tidur (Jam)", 1.0, 12.0, 6.0)
    physical = st.slider("Aktivitas Fisik (jam)", 0.0, 15.0, 0.0)
    alcohol = st.selectbox("Apakah Anda Meminum Alkohol ?", ["Tidak", "Ya"])


bmi = berat / ((tinggi / 100) ** 2)
gender_map = {"Pria": 0, "Perempuan": 1, "Lainnya": 2}
smoking_map = {"Tidak": 0, "Ya": 1}
alcohol_map = {"Tidak": 0, "Ya": 1}

gender = gender_map.get(gender, -1)  # -1 is a default if gender not in map
smoking = smoking_map.get(smoking, -1)
alcohol = alcohol_map.get(alcohol, -1)

data_baru = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'Coffee_Intake': coffee,
    'Caffeine_mg': caffeine,
    'Sleep_Hours': sleep,
    'BMI': bmi,
    'Physical_Activity_Hours': physical,
    'Smoking': smoking,
    'Alcohol_Consumption': alcohol
}])

model = tf.keras.models.load_model('model_100.keras')
preprocessor_loaded = joblib.load('preprocessor.joblib')

data_baru_prep = preprocessor_loaded.transform(data_baru)
hasil_prediksi = model.predict(data_baru_prep)

prediksi_tidur = hasil_prediksi[0]
prediksi_stres = hasil_prediksi[1]

indeks_tidur = np.argmax(prediksi_tidur, axis=1)[0]
indeks_stres = np.argmax(prediksi_stres, axis=1)[0]

# label_tidur = {0: "Poor (Buruk)", 1: "Fair (Cukup)", 2: "Good (Baik)", 3: "Excellent (Sangat Baik)"}
# label_stres = {0: "Low (Rendah)", 1: "Medium (Sedang)", 2: "High (Tinggi)"}

st.write("")
st.write("")
if st.button("Jalankan Prediksi", type="primary"):
    # Proses loading
    with st.spinner("Sedang memproses..."):
        time.sleep(1)

    col3, col4 = st.columns(2)
    with col3:
        if indeks_tidur==0:
            st.error("Kualitas Tidur Buruk")
        elif indeks_tidur==1:
            st.warning("Kualitas Tidur Cukup")
        elif indeks_tidur==2:
            st.info("Kualitas Tidur Baik")
        elif indeks_tidur==3:
            st.success("Kualitas Tidur Sangat Baik")

    with col4:
        if indeks_stres==0:
            st.success("Tingkat Stres Rendah")
        elif indeks_stres==1:
            st.warning("Tingkat Stres Sedang")
        elif indeks_stres==2:
            st.error("Tingkat Stres Tinggi")

    with st.expander("Lihat Detail Probabilitas"):
        st.write("Probabilitas Kualitas Tidur:", prediksi_tidur)
        st.write("Probabilitas Tingkat Stres:", prediksi_stres)

