import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import random

st.title("🌋 Visualisasi Prediksi Gempa Indonesia ")

st.markdown("""
Aplikasi ini menampilkan simulasi prediksi gempa di wilayah Indonesia**.
""")

# ============================
# Bagian 1: Prediksi Gempa Selanjutnya
# ============================
st.header("🔮 Prediksi Gempa Selanjutnya")

jumlah_prediksi = st.number_input(
    "Masukkan jumlah prediksi gempa:",
    min_value=1, max_value=10, value=3
)

if st.button("Tampilkan Prediksi Gempa"):
    lokasi = ["Sumatera", "Jawa", "Bali", "NTT", "NTB", "Sulawesi", "Papua", "Maluku", "Kalimantan"]
    df_prediksi = pd.DataFrame({
        "Wilayah": [random.choice(lokasi) for _ in range(jumlah_prediksi)],
        "Latitude": np.random.uniform(-10, 6, jumlah_prediksi),
        "Longitude": np.random.uniform(95, 141, jumlah_prediksi),
        "Kedalaman (km)": np.random.uniform(5, 200, jumlah_prediksi).round(1),
        "Magnitudo (Mw)": np.random.uniform(3.0, 7.0, jumlah_prediksi).round(2)
    })
    # Buat kolom radius agar tidak ada ekspresi di JSON
    df_prediksi["radius"] = df_prediksi["Magnitudo (Mw)"] * 10000

    st.dataframe(df_prediksi)

    # Layer PyDeck (tanpa ekspresi)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_prediksi,
        get_position=["Longitude", "Latitude"],
        get_radius="radius",
        get_fill_color=[255, 100, 50, 180],
        pickable=True
    )

    view_state = pdk.ViewState(latitude=-2, longitude=118, zoom=4.5)

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "📍 {Wilayah}\nMagnitudo: {Magnitudo (Mw)} Mw\nKedalaman: {Kedalaman (km)} km"}
    ))

# ============================
# Bagian 2: Prediksi Berdasarkan Input
# ============================
st.header("📍 Prediksi Magnitudo Berdasarkan Koordinat")

lat = st.number_input("Masukkan Latitude (-10 s.d 6):", min_value=-10.0, max_value=6.0, value=-2.0)
lon = st.number_input("Masukkan Longitude (95 s.d 141):", min_value=95.0, max_value=141.0, value=120.0)
depth = st.number_input("Masukkan Kedalaman (km):", min_value=0.0, max_value=700.0, value=50.0)

if st.button("Prediksi Magnitudo"):
    magnitude_pred = round(3 + np.random.rand() * (7 - 3) - (depth / 1000), 2)
    magnitude_pred = max(magnitude_pred, 2.5)
    st.success(f"🔎 Prediksi Magnitudo: **{magnitude_pred} Mw**")

    df_input = pd.DataFrame({
        "Latitude": [lat],
        "Longitude": [lon],
        "Magnitudo (Mw)": [magnitude_pred],
        "Kedalaman (km)": [depth],
        "radius": [magnitude_pred * 15000]
    })

    layer_input = pdk.Layer(
        "ScatterplotLayer",
        data=df_input,
        get_position=["Longitude", "Latitude"],
        get_radius="radius",
        get_fill_color=[255, 0, 0, 200],
        pickable=True
    )

    view_state_input = pdk.ViewState(latitude=lat, longitude=lon, zoom=5)

    st.pydeck_chart(pdk.Deck(
        layers=[layer_input],
        initial_view_state=view_state_input,
        tooltip={"text": "Koordinat: [{Latitude}, {Longitude}]\nMagnitudo: {Magnitudo (Mw)} Mw\nKedalaman: {Kedalaman (km)} km"}
    ))

st.markdown("---")

