import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from geopy.distance import geodesic
import folium
import streamlit as st
import tempfile

st.markdown("<h1 style='text-align: center; color: pink;'>PIHALENKKI</h1>", unsafe_allow_html=True)

# Tiedostot
gps_file = 'Location.csv'
accel_file = 'Linear Accelerometer.csv'

# Ladataan tiedot
gps_data = pd.read_csv(gps_file)
accelerometer_data = pd.read_csv(accel_file)

# Informaatio-osio
info_column = st.container()  # Luo säiliö informaatiota varten

# Kiihtyvyysdata
time = accelerometer_data['Time (s)']
acc_z = accelerometer_data['Z (m/s^2)']

# Suodatetaan kiihtyvyysdata
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

cutoff = 2.5  
fs = 1 / np.mean(np.diff(time))  
filtered_acc_z = butter_lowpass_filter(acc_z, cutoff, fs)

# Lasketaan askelmäärät
peaks, _ = find_peaks(filtered_acc_z, height=0.3)
askelmaara = len(peaks)

N = len(filtered_acc_z)
T = 1.0 / fs
yf = fft(filtered_acc_z)
xf = fftfreq(N, T)[:N//2]

amplitudes = 2.0 / N * np.abs(yf[:N // 2])
dominant_freq_index = np.argmax(amplitudes[1:]) + 1  # Poistetaan nollataajuus
dominant_freq = xf[dominant_freq_index]
askelmaara_fourier = dominant_freq * time.iloc[-1]

# GPS-data
lats = gps_data['Latitude (°)']
lons = gps_data['Longitude (°)']

total_distance = sum(geodesic((lats[i], lons[i]), (lats[i + 1], lons[i + 1])).meters for i in range(len(lats) - 1))
time_gps = gps_data['Time (s)']
duration = time_gps.iloc[-1] - time_gps.iloc[0]
average_speed = total_distance / duration if duration > 0 else 0  # m/s

askelpituus = total_distance / askelmaara if askelmaara > 0 else 0

# Informaatio-näkymä
with info_column:
    st.markdown(f"<p style='font-size: 16px; color: pink;'>Askelmäärä suodatetusta datasta: <span style='font-weight:bold;'>{askelmaara} askelta</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 16px; color: pink;'>Askelmäärä Fourier-analyysin perusteella: <span style='font-weight:bold;'>{askelmaara_fourier:.0f} askelta</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 16px; color: pink;'>Kuljettu matka: <span style='font-weight:bold;'>{total_distance:.2f} metriä</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 16px; color: pink;'>Keskinopeus: <span style='font-weight:bold;'>{average_speed:.2f} m/s</span></p>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size: 16px; color: pink;'>Askelpituus: <span style='font-weight:bold;'>{askelpituus:.2f} metriä</span></p>", unsafe_allow_html=True)

# Kuvanäkymä
image_column = st.container() 

# Z-komponentti
with image_column:
    st.markdown("<h2 style='text-align: center; color: pink;'>Suodatetun kiihtyvyysdatan Z-komponentti</h2>", unsafe_allow_html=True)
    plt.figure(figsize=(10, 6))
    plt.plot(time, filtered_acc_z, label='Suodatettu Z (m/s^2)')
    plt.title('Suodatettu Z-Kiihtyvyys')
    plt.xlabel('Aika (s)')
    plt.ylabel('Kiihtyvyys (m/s^2)')
    plt.legend()
    st.pyplot(plt)

# Tehospektritiheys
with image_column:
    st.markdown("<h2 style='text-align: center; color: pink;'>Tehospektritiheys</h2>", unsafe_allow_html=True)
    plt.figure(figsize=(10, 6))
    plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
    plt.title('Tehospektritiheys')
    plt.xlabel('Taajuus (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()
    st.pyplot(plt)

# Kartta
with image_column:
    st.markdown("<h2 style='text-align: center; color: pink;'>Kartta</h2>", unsafe_allow_html=True)
    m = folium.Map(location=[lats.mean(), lons.mean()], zoom_start=15)
    route = list(zip(lats, lons))
    folium.PolyLine(route, color="red", weight=2.5, opacity=1).add_to(m)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as f:
        m.save(f.name)
        f.seek(0)
        # Upota HTML-kartta Streamlit-sovellukseen
        st.components.v1.html(f.read(), height=500)




