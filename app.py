# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, SimpleRNN

st.title("📦 Prediksi Permintaan Produk Retail dengan Deep Learning")

# Upload file
uploaded_file = st.file_uploader("Unggah file CSV permintaan harian (format: date, demand):", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    st.subheader("Preview Data")
    st.dataframe(df.head())

    # Normalisasi
    scaler = MinMaxScaler()
    df['scaled'] = scaler.fit_transform(df[['demand']])

    # Buat time series window
    def make_seq(data, win=14):
        X, y = [], []
        for i in range(len(data) - win):
            X.append(data[i:i+win])
            y.append(data[i+win])
        return np.array(X), np.array(y)

    X, y = make_seq(df['scaled'].values)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Fungsi bangun model
    def build_model(model_type='LSTM'):
        model = Sequential()
        if model_type == 'LSTM':
            model.add(LSTM(50, input_shape=(X.shape[1], 1)))
        elif model_type == 'GRU':
            model.add(GRU(50, input_shape=(X.shape[1], 1)))
        elif model_type == 'RNN':
            model.add(SimpleRNN(50, input_shape=(X.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    # Training semua model
    with st.spinner('Melatih model...'):
        lstm_model = build_model('LSTM')
        lstm_model.fit(X, y, epochs=10, verbose=0)

        gru_model = build_model('GRU')
        gru_model.fit(X, y, epochs=10, verbose=0)

        rnn_model = build_model('RNN')
        rnn_model.fit(X, y, epochs=10, verbose=0)

    # Prediksi
    y_actual = scaler.inverse_transform(y.reshape(-1, 1))
    y_lstm = scaler.inverse_transform(lstm_model.predict(X))
    y_gru = scaler.inverse_transform(gru_model.predict(X))
    y_rnn = scaler.inverse_transform(rnn_model.predict(X))

    result_df = pd.DataFrame({
        'date': df.index[14:],
        'actual': y_actual.flatten(),
        'lstm_pred': y_lstm.flatten(),
        'gru_pred': y_gru.flatten(),
        'rnn_pred': y_rnn.flatten()
    })

    st.subheader("📈 Grafik Hasil Prediksi")
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(result_df['date'], result_df['actual'], label='Actual', linewidth=2)
    ax.plot(result_df['date'], result_df['lstm_pred'], label='LSTM')
    ax.plot(result_df['date'], result_df['gru_pred'], label='GRU')
    ax.plot(result_df['date'], result_df['rnn_pred'], label='RNN')
    ax.legend()
    st.pyplot(fig)

    # Evaluasi
    def evaluate(y_true, y_pred):
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return mae, rmse, mape

    eval_df = pd.DataFrame({
        'Model': ['LSTM', 'GRU', 'RNN'],
        'MAE': [
            evaluate(result_df['actual'], result_df['lstm_pred'])[0],
            evaluate(result_df['actual'], result_df['gru_pred'])[0],
            evaluate(result_df['actual'], result_df['rnn_pred'])[0]
        ],
        'RMSE': [
            evaluate(result_df['actual'], result_df['lstm_pred'])[1],
            evaluate(result_df['actual'], result_df['gru_pred'])[1],
            evaluate(result_df['actual'], result_df['rnn_pred'])[1]
        ],
        'MAPE (%)': [
            evaluate(result_df['actual'], result_df['lstm_pred'])[2],
            evaluate(result_df['actual'], result_df['gru_pred'])[2],
            evaluate(result_df['actual'], result_df['rnn_pred'])[2]
        ]
    })

    st.subheader("📊 Evaluasi Model")
    st.dataframe(eval_df.style.format("{:.2f}"))

    # Unduh hasil
    st.download_button("📥 Unduh Hasil Prediksi CSV", data=result_df.to_csv(index=False), file_name="hasil_prediksi.csv")
