import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import random

# Judul aplikasi
st.title("🌋 Visualisasi Prediksi Gempa Indonesia ")

st.markdown("""
Aplikasi ini menampilkan simulasi prediksi gempa di wilayah Indonesia berdasarkan input pengguna.

""")

# ============================
# Bagian 1: Prediksi Gempa Selanjutnya (Map)
# ============================
st.header("🔮 Prediksi Gempa Selanjutnya")

jumlah_prediksi = st.number_input(
    "Masukkan jumlah prediksi gempa:",
    min_value=1, max_value=10, value=3
)

if st.button("Tampilkan Prediksi Gempa di Peta"):
    # Data acak untuk simulasi
    lokasi = ["Sumatera", "Jawa", "Bali", "NTT", "NTB", "Sulawesi", "Papua", "Maluku", "Kalimantan"]
    df_prediksi = pd.DataFrame({
        "Wilayah": [random.choice(lokasi) for _ in range(jumlah_prediksi)],
        "Latitude": np.random.uniform(-10, 6, jumlah_prediksi),
        "Longitude": np.random.uniform(95, 141, jumlah_prediksi),
        "Kedalaman (km)": np.random.uniform(5, 200, jumlah_prediksi).round(1),
        "Magnitudo (Mw)": np.random.uniform(3.0, 7.0, jumlah_prediksi).round(2)
    })

    # Tampilkan tabel
    st.dataframe(df_prediksi)

    # Buat peta Indonesia
    map_pred = folium.Map(location=[-2, 118], zoom_start=5)

    # Tambahkan titik ke peta
    for _, row in df_prediksi.iterrows():
        color = "red" if row["Magnitudo (Mw)"] >= 6 else "orange" if row["Magnitudo (Mw)"] >= 4.5 else "blue"
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=row["Magnitudo (Mw)"],  # ukuran berdasarkan magnitudo
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=(
                f"<b>Wilayah:</b> {row['Wilayah']}<br>"
                f"<b>Magnitudo:</b> {row['Magnitudo (Mw)']} Mw<br>"
                f"<b>Kedalaman:</b> {row['Kedalaman (km)']} km"
            )
        ).add_to(map_pred)

    # Tampilkan peta
    st_data = st_folium(map_pred, width=700, height=450)

# ============================
# Bagian 2: Prediksi Magnitudo Berdasarkan Input
# ============================
st.header("📍 Prediksi Magnitudo Berdasarkan Koordinat")

lat = st.number_input("Masukkan Latitude (-10 s.d 6):", min_value=-10.0, max_value=6.0, value=-2.0)
lon = st.number_input("Masukkan Longitude (95 s.d 141):", min_value=95.0, max_value=141.0, value=120.0)
depth = st.number_input("Masukkan Kedalaman (km):", min_value=0.0, max_value=700.0, value=50.0)

if st.button("Prediksi Magnitudo"):
    # Simulasi model: magnitude naik dengan depth dan random noise
    magnitude_pred = round(3 + np.random.rand() * (7 - 3) - (depth / 1000), 2)
    magnitude_pred = max(magnitude_pred, 2.5)

    st.success(f"🔎 Prediksi Magnitudo: **{magnitude_pred} Mw**")

    # Peta tunggal lokasi input
    map_input = folium.Map(location=[lat, lon], zoom_start=6)
    folium.CircleMarker(
        location=[lat, lon],
        radius=magnitude_pred,
        color="red",
        fill=True,
        fill_opacity=0.7,
        popup=f"<b>Prediksi Magnitudo:</b> {magnitude_pred} Mw<br><b>Kedalaman:</b> {depth} km"
    ).add_to(map_input)

    st_folium(map_input, width=700, height=450)

# Footer
st.markdown("---")

