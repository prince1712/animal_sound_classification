import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from PIL import Image
from streamlit_option_menu import option_menu


st.set_page_config(page_title="Jungle Tune", page_icon="🐶", layout="wide")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")


@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model("custom_cnn_classifier_93_percent.keras")
    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, label_encoder = load_model_and_encoder()


SAMPLE_RATE = 22050
DURATION = 5
N_MELS = 128
HOP_LENGTH = 512
FIXED_LENGTH = int(SAMPLE_RATE * DURATION)

def get_mel_spectrogram(audio_data, sr=SAMPLE_RATE):
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    return librosa.power_to_db(mel_spec, ref=np.max)

def preprocess_audio(file_path):
    data, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    if len(data) > FIXED_LENGTH:
        data = data[:FIXED_LENGTH]
    else:
        data = np.pad(data, (0, max(0, FIXED_LENGTH - len(data))), "constant")
    spec = get_mel_spectrogram(data, sr)
    spec = np.expand_dims(spec, axis=-1)
    spec = np.expand_dims(spec, axis=0)
    return spec, data


with st.sidebar:
    st.image("logo animal.png", use_container_width=True)
    app_mode = option_menu(
        menu_title="Jungle Tune",
        options=["Home", "About Project", "Prediction", "Visualizations"],
        icons=["house", "info-circle", "mic", "bar-chart"],
        menu_icon="music-note-beamed",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "rgba(255,255,255,0.7)", "border-radius": "12px"},
            "icon": {"color": "#7E57C2", "font-size": "22px"},
            "nav-link": {"color": "#4A148C", "font-size": "18px", "font-weight": "500"},
            "nav-link-selected": {"background-color": "#E1BEE7", "color": "#311B92"},
        }
    )

if app_mode == "Home":
    
    st.markdown("""
        <div class='app-header'>
            <h1>🐶 Jungle Tune</h1>
            <h4>AI-Powered Animal Sound Classifier</h4>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("""
    <div class="dashboard-container">
        <div class="dashboard-card">
            <h3>🎵 Upload Sound</h3>
            <p>Upload animal sound clips and let AI identify the species.</p>
        </div>
        <div class="dashboard-card">
            <h3>🤖 Model Accuracy</h3>
            <h2>90%</h2>
            <p>Trained on 13 animal classes with 650+ samples.</p>
        </div>
        <div class="dashboard-card">
            <h3>🧠 Technology Used</h3>
            <p>Deep CNN with Mel-Spectrograms for precise sound recognition.</p>
        </div>
        <div class="dashboard-card">
            <h3>📊 Predictions</h3>
            <p>Shows confidence and class probabilities for each animal sound.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("""
    <div class='home-info'>
        <h2>🐘 Welcome to Jungle Tune!</h2>
        <p>Experience how artificial intelligence can recognize and classify animal voices with high accuracy.</p>
        <p>Simply navigate to the <b>Prediction</b> page to upload an audio clip and see the results.</p>
    </div>
    """, unsafe_allow_html=True)


elif app_mode == "About Project":
    st.markdown("""
    <div class='section'>
        <h2>📘 About the Project</h2>
        <p><b>Jungle Tune</b> is a Deep Learning based Animal Sound Classifier that identifies 13 animal species from 5-second sound clips.</p>
        <ul>
            <li>🐾 Uses Mel-Spectrograms to transform audio into visual data for CNN analysis.</li>
            <li>🎯 Achieves <b>93% accuracy</b> on 650+ audio samples.</li>
            <li>📂 Dataset includes: Lion, Bear, Cat, Chicken, Cow, Dog, Dolphin, Donkey, Elephant, Frog, Horse, Monkey, Sheep.</li>
        </ul>
        <h3>🔧 Tools & Technologies</h3>
        <p>TensorFlow • Librosa • NumPy • Streamlit • Matplotlib</p>
    </div>
    """, unsafe_allow_html=True)


elif app_mode == "Prediction":
    st.markdown("<h2 class='section-title'>🎧 Animal Sound Prediction</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an animal sound file (.wav format)", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        if st.button("🔍 Predict"):
            with open("temp.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
            spec, data = preprocess_audio("temp.wav")

            pred = model.predict(spec)
            idx = np.argmax(pred, axis=1)[0]
            conf = np.max(pred) * 100
            label = label_encoder.classes_[idx]

            st.markdown(f"""
            <div class='result-card'>
                <h3>🐾 Predicted Animal:</h3>
                <h2>{label}</h2>
                <p>Confidence: <b>{conf:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            # Probability chart
            prob_dict = {label_encoder.classes_[i]: float(pred[0][i]) for i in range(len(label_encoder.classes_))}
            st.bar_chart(prob_dict)
    else:
        st.info("⬆️ Upload a sound file to start prediction.")


elif app_mode == "Visualizations":
    st.markdown("<h2 class='section-title'>📊 Audio Visualizations</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an animal sound file (.wav format)", type=["wav", "mp3"])

    if uploaded_file is not None:
        y, sr = librosa.load(uploaded_file, sr=None)
        st.audio(uploaded_file)

        # Waveform
        fig_wave, ax_wave = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax_wave)
        ax_wave.set(title="Waveform")
        st.pyplot(fig_wave)

        # Mel-Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig_mel, ax_mel = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", fmax=8000, ax=ax_mel)
        fig_mel.colorbar(img, ax=ax_mel, format="%+2.0f dB")
        ax_mel.set(title="Mel-Spectrogram")
        st.pyplot(fig_mel)
    else:
        st.info("🎵 Upload an audio file to view its waveform and spectrogram.")


st.markdown("""
<hr class='divider'>
<div class='footer-text'>
    Made with 💜 using Streamlit | Jungle Tune © 2025
</div>
""", unsafe_allow_html=True)
