import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np
import random

# Judul Aplikasi
st.title("🌋 Aplikasi Prediksi dan Deteksi Gempa Indonesia")
st.write("Aplikasi ini memprediksi lokasi dan magnitudo gempa yang mungkin terjadi berdasarkan data seismik.")

st.subheader("1️⃣ Prediksi Peta Gempa Selanjutnya")

# Input jumlah prediksi
jumlah_prediksi = st.number_input("Masukkan jumlah prediksi gempa:", min_value=1, max_value=50, value=5)

if st.button("Prediksi Peta Gempa"):
    # Contoh simulasi titik gempa acak di Indonesia
    lats = np.random.uniform(-11.0, 6.0, jumlah_prediksi)
    lons = np.random.uniform(95.0, 141.0, jumlah_prediksi)
    mags = np.random.uniform(3.0, 7.0, jumlah_prediksi)

    data = pd.DataFrame({'Latitude': lats, 'Longitude': lons, 'Magnitude': mags})

    # Membuat peta
    m = folium.Map(location=[-2.5, 118.0], zoom_start=5)
    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=row['Magnitude'],  # radius disesuaikan magnitudo
            popup=f"Magnitude: {row['Magnitude']:.2f}",
            color='red' if row['Magnitude'] > 5 else 'orange',
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    st_folium(m, width=700, height=450)

st.subheader("2️⃣ Prediksi Magnitudo Berdasarkan Lokasi")

# Input manual
lat = st.number_input("Masukkan Latitude:", value=-6.2)
lon = st.number_input("Masukkan Longitude:", value=106.8)
depth = st.number_input("Masukkan Depth (km):", value=10.0)

if st.button("Prediksi Magnitudo"):
    # Simulasi prediksi (nanti bisa diganti model asli)
    magnitude_pred = 3.5 + 0.02 * depth + random.uniform(-0.5, 0.5)
    st.success(f"🔮 Perkiraan Magnitudo: **{magnitude_pred:.2f}**")
