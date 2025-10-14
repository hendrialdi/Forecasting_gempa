"""
Earthquake Forecast Demo (Streamlit + Folium)

Files in this single-file demo:
- app.py (this file): Streamlit app using folium to visualize earthquake points in Indonesia
- requirements.txt (see README below)

README / Deployment:
1. Create a new GitHub repository (e.g. earthquake-forecast-app) and add this app.py file.
2. Add a requirements.txt containing: streamlit
   folium
   streamlit-folium
   pandas
   numpy
3. Commit and push to GitHub.
4. Deploy on Streamlit Community Cloud:
   - Go to https://streamlit.io/cloud and sign in with your GitHub account.
   - Create a new app and point it to the repository and branch containing this app.py.
   - Streamlit will install packages from requirements.txt and run `streamlit run app.py`.

IMPORTANT DISCLAIMER:
This application is a demonstration/prototype only. The "forecast" implemented here is a simple heuristic scoring function for visualization and educational purposes — it is NOT a validated earthquake prediction model and MUST NOT be used for real-world decision-making, warnings, or safety-critical operations.

Usage (app features):
- Input one earthquake candidate (magnitude, latitude, longitude, depth) and see a forecast score (0-100%)
- Optionally upload a CSV of multiple points (columns: lat, lon, depth, mag) to visualize many points
- Map centers over Indonesia by default

"""

import streamlit as st
from streamlit_folium import st_folium
import folium
import pandas as pd
import numpy as np
import math
import random

st.set_page_config(page_title="Earthquake Forecast Demo", layout="wide")

# --- Helper functions ---

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def simple_forecast_score(mag, depth, lat=None, lon=None):
    """
    Very simple heuristic scoring function (DEMONSTRATION ONLY):
    - Higher magnitude increases score
    - Shallower depth increases score
    - Small random noise to break ties
    Returns percentage 0-100
    """
    # Normalize magnitude around 4.5
    mag_term = (mag - 4.5) * 1.3
    # Depth penalty (shallower -> higher score). Depth in km. typical shallow quakes <70
    depth_term = - (depth / 100.0)
    # optional latitude/longitude factor to slightly increase scores inside Indonesia bounds
    indo_bonus = 0.0
    if lat is not None and lon is not None:
        # rough bounding box for Indonesia
        if -15 <= lat <= 6 and 95 <= lon <= 141:
            indo_bonus = 0.3
    noise = random.uniform(-0.2, 0.2)
    raw = mag_term + depth_term + indo_bonus + noise
    prob = sigmoid(raw)
    percent = max(0.0, min(100.0, prob * 100))
    return round(percent, 1)


# --- UI ---

st.title("Earthquake Forecast — Demo (Indonesia)")
st.markdown("""
Masukkan parameter gempa (magnitudo, koordinat lat/lon, kedalaman) atau unggah CSV berisi beberapa titik.

**Catatan:** ini hanya demo visualisasi dan *bukan* model prediksi gempa yang valid.
""")

with st.sidebar:
    st.header("Input")
    mode = st.radio("Mode input:", ["Single point", "Upload CSV"], index=0)
    if mode == "Single point":
        mag = st.number_input("Magnitude (Mw)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        lat = st.number_input("Latitude", value=-2.5, format="%f")
        lon = st.number_input("Longitude", value=118.0, format="%f")
        depth = st.number_input("Depth (km)", min_value=0.0, max_value=700.0, value=30.0)
        show_score = st.checkbox("Tampilkan skor forecast", value=True)
        submit = st.button("Tampilkan pada peta")
    else:
        uploaded = st.file_uploader("Unggah CSV (lat,lon,depth,mag)", type=["csv"])
        csv_example = "lat,lon,depth,mag\n-6.2,106.8,10,4.7\n-2.5,118.0,30,5.1\n-7.5,110.4,45,6.0"
        st.caption("Contoh isi CSV:\n" + csv_example)
        submit = st.button("Unggah dan tampilkan CSV")

# Default center => Indonesia (approx)
center_lat, center_lon = -2.5, 118.0

# Initialize folium map
m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="OpenStreetMap")

# Add a simple basemap layer control
folium.TileLayer('Stamen Terrain').add_to(m)
folium.LayerControl().add_to(m)

# If single point and submitted
if mode == "Single point" and submit:
    score = simple_forecast_score(mag, depth, lat, lon)
    popup_text = f"Mag: {mag}, Depth: {depth} km\nForecast score: {score}% (demo)"
    # Color by score: green low, orange mid, red high
    if score >= 66:
        color = 'red'
    elif score >= 33:
        color = 'orange'
    else:
        color = 'green'
    folium.CircleMarker(location=[lat, lon], radius=10, color=color, fill=True, fill_opacity=0.7,
                        popup=popup_text).add_to(m)
    st.subheader("Hasil Forecast (Demo)")
    st.markdown(f"**Skor:** {score}% — interpretasi: hanya ilustrasi")

# If CSV upload
if mode == "Upload CSV" and submit:
    if uploaded is None:
        st.warning("Silakan unggah file CSV terlebih dahulu.")
    else:
        try:
            df = pd.read_csv(uploaded)
            # Expect columns: lat, lon, depth, mag (tolerant)
            colnames = [c.lower().strip() for c in df.columns]
            # find columns
            def find_col(options):
                for o in options:
                    if o in colnames:
                        return df.columns[colnames.index(o)]
                return None
            lat_col = find_col(['lat','latitude'])
            lon_col = find_col(['lon','longitude','lng'])
            depth_col = find_col(['depth'])
            mag_col = find_col(['mag','magnitude','mw'])
            if not lat_col or not lon_col or not depth_col or not mag_col:
                st.error('CSV harus berisi kolom: lat, lon, depth, mag (nama kolom case-insensitive)')
            else:
                points = []
                for _, row in df.iterrows():
                    rlat = float(row[lat_col])
                    rlon = float(row[lon_col])
                    rdepth = float(row[depth_col])
                    rmag = float(row[mag_col])
                    score = simple_forecast_score(rmag, rdepth, rlat, rlon)
                    points.append((rlat, rlon, rdepth, rmag, score))
                    popup = f"Mag: {rmag}, Depth: {rdepth} km\nScore: {score}%"
                    if score >= 66:
                        color = 'red'
                    elif score >= 33:
                        color = 'orange'
                    else:
                        color = 'green'
                    folium.CircleMarker(location=[rlat, rlon], radius=6, color=color, fill=True,
                                        fill_opacity=0.7, popup=popup).add_to(m)
                st.success(f"Ditampilkan {len(points)} titik dari CSV (demo)")
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")

# Always show map in main area
st.subheader("Peta — Visualisasi Titik Gempa (Indonesia)")
with st.expander("Opsi peta"):
    st.write("Gunakan zoom dan pan pada peta. Klik marker untuk melihat detail (demo).")

# Render folium map in Streamlit
st_data = st_folium(m, width=900, height=600)

# Footer / tips
st.markdown("---")
st.caption("Aplikasi deteksi gempa")

# If user wants to test multiple random demo points
if st.button('Generate contoh acak 20 titik di Indonesia'):
    for i in range(20):
        rlat = random.uniform(-10.0, 5.0)
        rlon = random.uniform(95.0, 141.0)
        rdepth = random.uniform(5, 300)
        rmag = random.uniform(4.0, 7.5)
        score = simple_forecast_score(rmag, rdepth, rlat, rlon)
        popup = f"Mag: {round(rmag,2)}, Depth: {int(rdepth)} km\nScore: {score}%"
        if score >= 66:
            color = 'red'
        elif score >= 33:
            color = 'orange'
        else:
            color = 'green'
        folium.CircleMarker(location=[rlat, rlon], radius=6, color=color, fill=True,
                            fill_opacity=0.7, popup=popup).add_to(m)
    st.experimental_rerun()
