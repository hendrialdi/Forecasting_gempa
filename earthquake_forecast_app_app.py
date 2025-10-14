
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import learn2learn as l2l
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import folium
from geopy.geocoders import Nominatim
import time
import torch.optim as optim
from streamlit_folium import st_folium

st.title("Aplikasi Prediksi Gempa Bumi Meta-Learning")
st.write("Aplikasi ini menggunakan model Meta-Learning (MAML) untuk memprediksi karakteristik gempa bumi di masa depan.")

torch.set_float32_matmul_precision('high')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

WIN = 10
N_TASKS = 6
# CSV_PATH = "/content/gabungan_2010-2025_utc.csv" # Placeholder path - user needs to ensure this file is accessible

# =========================
# Data Loading and Preprocessing Function (Steps 2-5)
# Modified to accept uploaded file object
# =========================
@st.cache_resource(show_spinner="Loading and preprocessing data...")
def load_and_preprocess_data(uploaded_file, use_cols, win=10, n_tasks=6):
    if uploaded_file is not None:
        try:
            # Read the uploaded file into a pandas DataFrame
            df = pd.read_csv(uploaded_file)
            df = df.dropna(subset=use_cols).reset_index(drop=True)
            raw = df[use_cols].astype(np.float32).values
            scaler = MinMaxScaler()
            data = scaler.fit_transform(raw).astype(np.float32)

            def make_sequences(arr, win):
                X, y = [], []
                for i in range(len(arr)-win):
                    X.append(arr[i:i+win])
                    y.append(arr[i+win])
                return np.asarray(X, np.float32), np.asarray(y, np.float32)

            X_seq, y = make_sequences(data, win)

            def seq_to_meta(X_seq):
                mean_ = X_seq.mean(axis=1)
                std_  = X_seq.std(axis=1)
                last_ = X_seq[:, -1, :]
                return np.concatenate([mean_, std_, last_], axis=1).astype(np.float32)

            X_meta = seq_to_meta(X_seq)

            def split_tasks(X_seq, X_meta, y, n_tasks):
                N = len(X_seq)
                size = N // n_tasks
                tasks = []
                for t in range(n_tasks):
                    s, e = t*size, (t+1)*size if t < n_tasks-1 else N
                    tasks.append((
                        X_seq[s:e],
                        X_meta[s:e],
                        y[s:e],
                    ))
                return tasks

            tasks = split_tasks(X_seq, X_meta, y, n_tasks=n_tasks)

            return data, scaler, tasks, X_seq.shape[2], X_meta.shape[1], y.shape[1]

        except Exception as e:
            st.error(f"An error occurred during data loading and preprocessing: {e}")
            return None, None, None, None, None, None
    else:
        return None, None, None, None, None, None


# =========================
# Step 7: Loss, Optimizer, dan MAML
# =========================

# -------------------------------
# 1. Model LSTM + Transformer + Meta Features (PyTorch)
# -------------------------------
class LSTMTransformerNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, meta_dim=12, output_dim=4, win=10):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm_fc = nn.Linear(hidden_dim, 32)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)
        self.trans_fc = nn.Linear(input_dim, 32)
        self.meta_fc = nn.Linear(meta_dim, 32)
        self.fusion_fc1 = nn.Linear(32+32+32, 64)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(64, output_dim)

    def forward(self, x_seq, x_meta):
        lstm_out, _ = self.lstm(x_seq)
        lstm_out = self.lstm_fc(lstm_out[:, -1, :])
        trans_out = self.transformer_encoder(x_seq)
        trans_out = self.trans_fc(trans_out.mean(dim=1))
        meta_out = self.meta_fc(x_meta)
        fusion = torch.cat([lstm_out, trans_out, meta_out], dim=1)
        fusion = self.fusion_fc1(fusion)
        fusion = self.dropout(fusion)
        return self.out(fusion)

# -------------------------------
# 2. Hybrid Loss
# -------------------------------
def hybrid_loss(pred, target):
    mse = nn.MSELoss()(pred, target)
    mae = nn.L1Loss()(pred, target)
    return mse + 0.5 * mae

# -------------------------------
# 4. Convert numpy tasks → tensor tasks
# -------------------------------
def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32, device=device)

# -------------------------------
# 5. Time-based split (support=masa lalu, query=masa depan)
# -------------------------------
def time_based_split(Xs, Xm, Ys, ratio=0.7):
    k = int(len(Xs) * ratio)
    return (
        Xs[:k], Xm[:k], Ys[:k],
        Xs[k:], Xm[k:], Ys[k:]
    )

# -------------------------------
# 8. MAML Training Loop Function
# -------------------------------
@st.cache_resource(show_spinner="Training MAML model...")
def train_maml_model(model_params, train_params, tensor_tasks):
    input_dim, hidden_dim, meta_dim, output_dim, win = model_params
    maml_lr, optimizer_lr, weight_decay, inner_steps, clip_norm, epochs = train_params

    model = LSTMTransformerNet(input_dim, hidden_dim, meta_dim, output_dim, win).to(device)
    maml_model = l2l.algorithms.MAML(model, lr=maml_lr, first_order=True)
    optimizer = optim.Adam(maml_model.parameters(), lr=optimizer_lr, weight_decay=weight_decay)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(epochs):
        meta_loss = 0.0
        task_count = 0

        for Xs, Xm, Ys in tensor_tasks:
            learner = maml_model.clone()
            Xs_sup, Xm_sup, Ys_sup, Xs_qry, Xm_qry, Ys_qry = time_based_split(Xs, Xm, Ys, ratio=0.7)

            for step in range(inner_steps):
                pred = learner(Xs_sup, Xm_sup)
                loss = hybrid_loss(pred, Ys_sup)
                learner.adapt(loss)

            pred_q = learner(Xs_qry, Xm_qry)
            loss_q = hybrid_loss(pred_q, Ys_qry)
            meta_loss += loss_q
            task_count += 1

        optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(maml_model.parameters(), clip_norm)
        optimizer.step()

        avg_meta_loss = meta_loss.item() / task_count if task_count > 0 else 0
        status_text.text(f"Epoch {epoch+1}/{epochs}, Average Meta-Loss: {avg_meta_loss:.4f}")
        progress_bar.progress((epoch + 1) / epochs)

    st.success("MAML training complete.")
    return maml_model

# =========================
# 9. Evaluation Function
# =========================
def evaluate_on_task(maml_model, Xs, Xm, Ys, scaler, adapt_ratio=0.8, inner_steps=5):
    Xs_t = torch.tensor(Xs, dtype=torch.float32, device=device)
    Xm_t = torch.tensor(Xm, dtype=torch.float32, device=device)
    Ys_t = torch.tensor(Ys, dtype=torch.float32, device=device)

    n = len(Xs_t)
    cut = int(adapt_ratio * n)

    learner = maml_model.clone()
    for _ in range(inner_steps):
        pred = learner(Xs_t[:cut], Xm_t[:cut])
        loss = hybrid_loss(pred, Ys_t[:cut])
        learner.adapt(loss)

    with torch.no_grad():
        pred = learner(Xs_t[cut:], Xm_t[cut:]).cpu().numpy()
        true = Ys_t[cut:].cpu().numpy()

    inv_pred = scaler.inverse_transform(pred)
    inv_true = scaler.inverse_transform(true)

    mae = mean_absolute_error(inv_true, inv_pred)
    rmse = np.sqrt(mean_squared_error(inv_true, inv_pred))
    return mae, rmse, inv_true, inv_pred

# =========================
# 4. Meta-Features Function (already defined in load_and_preprocess_data, but needed for roll_forecast and predict_magnitude)
# =========================
def seq_to_meta(X_seq):
    mean_ = X_seq.mean(axis=1)
    std_  = X_seq.std(axis=1)
    last_ = X_seq[:, -1, :]
    return np.concatenate([mean_, std_, last_], axis=1).astype(np.float32)

# =========================
# 9. Rolling Multi-Step Forecast Function
# =========================
def roll_forecast(maml_model, X_seq_last, scaler, steps=100, win=WIN):
    seq = X_seq_last.copy().astype(np.float32)
    preds_norm = []
    learner = maml_model.clone()

    for t in range(steps):
        meta = seq_to_meta(seq[None, ...])
        x_seq_t = torch.tensor(seq[None, ...], dtype=torch.float32, device=device)
        x_meta_t = torch.tensor(meta, dtype=torch.float32, device=device)

        with torch.no_grad():
            yhat = learner(x_seq_t, x_meta_t).cpu().numpy()[0]
        preds_norm.append(yhat)
        seq = np.vstack([seq[1:], yhat])

    preds_norm = np.array(preds_norm, np.float32)
    preds_orig = scaler.inverse_transform(preds_norm)
    return preds_orig

# =========================
# 10. Geocoding Function
# =========================
geolocator = Nominatim(user_agent="eq_predictor_app")

@st.cache_data(show_spinner="Getting location name...")
def get_location_name(lat, lon):
    try:
        loc = geolocator.reverse((lat, lon), language="id", timeout=10)
        if loc and "address" in loc.raw:
            addr = loc.raw["address"]
            return (addr.get("village") or
                    addr.get("county") or
                    addr.get("state") or
                    addr.get("country") or
                    "Lokasi tidak dikenal")
        else:
            return "Lokasi tidak ditemukan"
    except Exception as e:
        return f"Error Geocoding: {e}"

# =========================
# 11. Predict Future Earthquakes Function
# =========================
def predict_future_eq(maml_model, last_window, scaler, n_steps=10, win=WIN):
    future_pred = roll_forecast(maml_model, last_window, scaler, steps=n_steps, win=win)
    df_future = pd.DataFrame(future_pred, columns=["Magnitude", "Latitude", "Longitude", "Depth (km)"])

    st.info("Mencari nama lokasi untuk prediksi...")
    location_progress = st.progress(0)
    lokasi_list = []
    for i, (_, row) in enumerate(df_future.iterrows()):
        lokasi_list.append(get_location_name(row["Latitude"], row["Longitude"]))
        location_progress.progress((i + 1) / len(df_future))

    df_future["Lokasi"] = lokasi_list
    st.success("Nama lokasi ditemukan.")
    return df_future

# =========================
# 12. Plot on Map Function
# =========================
def plot_on_map(df_future, map_center=[-2.5, 118], zoom_start=5):
    fmap = folium.Map(location=map_center, zoom_start=zoom_start)

    for _, row in df_future.iterrows():
        lat, lon, mag, depth = (float(row["Latitude"]),
                                float(row["Longitude"]),
                                float(row["Magnitude"]),
                                float(row["Depth (km)"]))
        lokasi = row["Lokasi"]

        popup_info = (f"<b>Magnitude:</b> {mag:.2f}<br>"
                      f"<b>Latitude:</b> {lat:.2f}<br>"
                      f"<b>Longitude:</b> {lon:.2f}<br>"
                      f"<b>Depth:</b> {depth:.1f} km<br>"
                      f"<b>Lokasi:</b> {lokasi}")

        if depth < 50:
            color = "red"
        elif depth < 100:
            color = "orange"
        else:
            color = "blue"

        radius = max(1, mag * 1.5)

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_info, max_width=250)
        ).add_to(fmap)

    legend_html = """
     <div style="
     position: fixed;
     bottom: 50px; left: 50px; width: 200px; height: 140px;
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; padding: 10px;">
     <b>Legenda Kedalaman</b><br>
     <i style="background:red; width:10px; height:10px;
        border-radius:50%; display:inline-block;"></i> Dangkal (&lt;50 km)<br>
     <i style="background:orange; width:10px; height:10px;
        border-radius:50%; display:inline-block;"></i> Menengah (50–100 km)<br>
     <i style="background:blue; width:10px; height:10px;
        border-radius:50%; display:inline-block;"></i> Dalam (&gt;100 km)
     </div>
     """
    return fmap, legend_html

# =========================
# 13. Single Point Prediction Function
# =========================
def predict_magnitude(maml_model, data, scaler, lat, lon, depth, win=WIN):
    row = np.array([[0.0, lat, lon, depth]], dtype=np.float32)
    row_norm = scaler.transform(row)

    if len(data) < win:
         st.error(f"Data is too short ({len(data)} samples) to form a window of size {win}.")
         return None
    seq = np.vstack([data[-(win-1):], row_norm])
    meta = seq_to_meta(seq[None, ...])

    x_seq_t = torch.tensor(seq[None, ...], dtype=torch.float32, device=device)
    x_meta_t = torch.tensor(meta, dtype=torch.float32, device=device)

    with torch.no_grad():
        pred = maml_model(x_seq_t, x_meta_t).cpu().numpy()[0]

    pred_orig = scaler.inverse_transform(pred.reshape(1, -1))[0]
    return pred_orig[0]

# =========================
# Streamlit App Interface
# =========================

st.sidebar.header("Unggah File Data Gempa")
uploaded_file = st.sidebar.file_uploader("Unggah file CSV Anda", type=["csv"])

use_cols = ["Magnitude", "Latitude", "Longitude", "Depth (km)"]
data, scaler, tasks, F_SEQ, F_META, F_OUT = load_and_preprocess_data(uploaded_file, use_cols, WIN, N_TASKS)


if data is not None and scaler is not None and tasks is not None:
    # Model and Training Parameters
    model_params = (F_SEQ, 64, F_META, F_OUT, WIN)
    train_params = (0.01, 1e-3, 1e-5, 5, 1.0, 10) # maml_lr, optimizer_lr, weight_decay, inner_steps, clip_norm, epochs

    # Train MAML model if not already trained
    # Convert tasks to tensor_tasks only once for training
    @st.cache_resource(show_spinner="Converting data to tensors...")
    def convert_tasks_to_tensors(tasks):
        tensor_tasks = []
        for Xs, Xm, Ys in tasks:
            tensor_tasks.append((
                to_tensor(Xs),
                to_tensor(Xm),
                to_tensor(Ys),
            ))
        return tensor_tasks

    tensor_tasks = convert_tasks_to_tensors(tasks)

    maml = train_maml_model(model_params, train_params, tensor_tasks)

    if maml is not None: # Check if training was successful
        # =========================
        # Display Evaluation Results (Step 9)
        # =========================
        st.subheader("Evaluasi Per Task (pada skala asli)")
        st.write("Metrik MAE dan RMSE untuk setiap task pada data holdout:")
        evaluation_results = []
        for i, (Xs, Xm, Ys) in enumerate(tasks, 1):
            # Check if Xs, Xm, Ys are not None and have enough samples
            if Xs is not None and Xm is not None and Ys is not None and len(Xs) > WIN:
                 mae, rmse, tru, prd = evaluate_on_task(maml, Xs, Xm, Ys, scaler)
                 evaluation_results.append({"Task": i, "MAE": mae, "RMSE": rmse})
            else:
                 evaluation_results.append({"Task": i, "MAE": "N/A", "RMSE": "N/A"})


        eval_df = pd.DataFrame(evaluation_results)
        st.dataframe(eval_df)

        # =========================
        # Visualize Holdout (Step 9)
        # =========================
        st.subheader("Visualisasi Holdout (Actual vs Predicted)")
        st.write("Perbandingan nilai aktual dan prediksi untuk 300 sampel pertama dari Task 1:")
        # Check if tasks[0] is valid before evaluating
        if tasks and len(tasks[0][0]) > WIN:
            mae, rmse, tru, prd = evaluate_on_task(maml, *tasks[0], scaler, adapt_ratio=0.8)

            labels = ["Magnitude", "Latitude", "Longitude", "Depth (km)"]

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            idx = slice(0, min(300, len(tru)))

            for i in range(4):
                axes[i].plot(tru[idx, i], marker='o', markersize=3, label='Aktual', alpha=0.8)
                plt.plot(prd[idx, i], marker='x', markersize=3, label='Prediksi', alpha=0.8)
                axes[i].set_title(f"{labels[i]}")
                axes[i].set_xlabel("Sample Index")
                axes[i].set_ylabel(labels[i])
                axes[i].grid(True, linestyle="--", alpha=0.6)
                axes[i].legend()

            plt.suptitle(f"Task 1 Holdout (first {idx.stop} samples) — MAE={mae:.4f}, RMSE={rmse:.4f}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            st.pyplot(fig)
            plt.close(fig) # Close the figure to prevent memory issues
        else:
            st.warning("Tidak cukup data di Task 1 untuk visualisasi holdout.")


        # =========================
        # Rolling Multi-Step Forecast (Step 9)
        # =========================
        st.subheader("Rolling Multi-Step Forecast ke Masa Depan")
        st.write("Prediksi karakteristik gempa bumi untuk beberapa langkah ke depan berdasarkan data historis terakhir.")

        n_steps_forecast = st.slider("Jumlah Langkah Prediksi ke Depan:", 10, 300, 100, key="forecast_steps")

        if st.button("Lakukan Prediksi Rolling Forecast", key="forecast_button"):
            # Check if data is long enough for a window
            if data is not None and len(data) >= WIN:
                st.info(f"Melakukan prediksi {n_steps_forecast} langkah ke depan...")
                last_window = data[-WIN:]
                future_pred = roll_forecast(maml, last_window, scaler, steps=n_steps_forecast, win=WIN)
                st.success("Prediksi Rolling Forecast Selesai.")

                fig_forecast, axes_forecast = plt.subplots(2, 2, figsize=(12, 8))
                axes_forecast = axes_forecast.flatten()
                labels = ["Magnitude", "Latitude", "Longitude", "Depth (km)"]

                for i, name in enumerate(labels):
                    axes_forecast[i].plot(future_pred[:, i], marker='x', label="Prediksi")
                    axes_forecast[i].set_title(f"Forecast ke Depan: {name}")
                    axes_forecast[i].set_xlabel("Step ke depan")
                    axes_forecast[i].set_ylabel(name)
                    axes_forecast[i].grid(True, linestyle="--", alpha=0.6)
                    axes_forecast[i].legend()

                plt.suptitle(f"Rolling Forecast {n_steps_forecast} Langkah ke Depan", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                st.pyplot(fig_forecast)
                plt.close(fig_forecast)

                fig_mag_forecast, ax_mag_forecast = plt.subplots(figsize=(10,4))
                ax_mag_forecast.plot(future_pred[:, 0], marker='o', markersize=4, label="Prediksi Magnitude")
                ax_mag_forecast.set_title("Forecast Magnitude ke Depan")
                ax_mag_forecast.set_xlabel("Step ke depan")
                ax_mag_forecast.set_ylabel("Magnitude (skala asli)")
                ax_mag_forecast.grid(True, linestyle="--", alpha=0.6)
                ax_mag_forecast.legend()
                st.pyplot(fig_mag_forecast)
                plt.close(fig_mag_forecast)
            else:
                st.warning("Tidak cukup data untuk melakukan rolling forecast. Harap unggah file CSV dengan data yang memadai.")


        # =========================
        # Interactive Single Point Prediction (Step 9)
        # =========================
        st.subheader("Prediksi Magnitude pada Lokasi Tertentu")
        st.write("Masukkan koordinat dan kedalaman untuk memprediksi Magnitude di lokasi tersebut.")

        lat_input = st.number_input("Latitude:", value=-2.5, format="%.4f", step=0.1, key="single_lat")
        lon_input = st.number_input("Longitude:", value=118.0, format="%.4f", step=0.1, key="single_lon")
        depth_input = st.number_input("Kedalaman (km):", value=50.0, format="%.1f", step=1.0, key="single_depth")

        if st.button("Prediksi Magnitude", key="single_predict_button"):
             # Check if data is long enough for a window before predicting
             if data is not None and len(data) >= WIN:
                st.info(f"Memprediksi Magnitude untuk lokasi ({lat_input:.2f}, {lon_input:.2f}), depth {depth_input:.1f} km...")
                mag_pred = predict_magnitude(maml, data, scaler, lat_input, lon_input, depth_input, win=WIN)

                if mag_pred is not None:
                    st.write(f"**Prediksi Magnitude:** {mag_pred:.2f}")
                    location_name = get_location_name(lat_input, lon_input)
                    st.write(f"**Perkiraan Lokasi:** {location_name}")
                st.success("Prediksi Magnitude Selesai.")
             else:
                 st.warning("Tidak cukup data untuk melakukan prediksi. Harap unggah file CSV dengan data yang memadai.")


        # =========================
        # Forecast Future Earthquakes and Plot on Map (Step 9)
        # =========================
        st.subheader("Prediksi Gempa Bumi Masa Depan dan Plot di Peta")
        st.write("Memprediksi lokasi dan karakteristik gempa bumi di masa depan dan menampilkannya di peta.")

        n_steps_map = st.slider("Jumlah Gempa yang Diprediksi untuk Peta:", 1, 50, 10, key="map_steps")

        if st.button("Prediksi dan Plot di Peta", key="map_button"):
             # Check if data is long enough for a window before predicting
             if data is not None and len(data) >= WIN:
                st.info(f"Memprediksi {n_steps_map} gempa ke depan dan memplot di peta...")
                last_window = data[-WIN:]
                df_future = predict_future_eq(maml, last_window, scaler, n_steps=n_steps_map, win=WIN)
                st.success("Prediksi dan Plot di Peta Selesai.")

                st.write(f"\nPrediksi {n_steps_map} gempa ke depan:")
                st.dataframe(df_future)

                fmap, legend_html = plot_on_map(df_future, map_center=[-2.5, 118], zoom_start=5)

                st_map = folium.Map(location=[-2.5, 118], zoom_start=5)
                for name, layer in fmap._children.items():
                     st_map.add_child(layer, name=name)

                st.markdown(legend_html, unsafe_allow_html=True)
                st_folium(st_map, width=700, height=500)
             else:
                 st.warning("Tidak cukup data untuk memprediksi dan memplot di peta. Harap unggah file CSV dengan data yang memadai.")


else:
    st.info("Unggah file CSV data gempa bumi untuk memulai.")
