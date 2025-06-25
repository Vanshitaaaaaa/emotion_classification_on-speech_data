import streamlit as st
import torch
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from model import EnhancedCNNLSTM

# --------------------------
# Constants and Configuration
# --------------------------
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
MODEL_PATH = "best_model.pth"

# --------------------------
# Model Loading
# --------------------------
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedCNNLSTM(num_classes=len(EMOTION_LABELS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# --------------------------
# Feature Extraction
# --------------------------
def extract_features(data, sample_rate):
    features = []
    
    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data))
    features.append(zcr)
    
    # Chroma STFT
    stft = np.abs(librosa.stft(data))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate))
    features.extend(chroma)
    
    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=20).T, axis=0)
    features.extend(mfcc)
    
    # RMS
    rms = np.mean(librosa.feature.rms(y=data))
    features.append(rms)
    
    # Mel Spectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate))
    features.append(mel)
    
    return np.array(features)

def process_audio(file):
    y, sr = librosa.load(file, duration=2.5, offset=0.6)
    features = extract_features(y, sr)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.reshape(1, -1))
    return features_scaled.reshape(1, 1, 9, 18)

# --------------------------
# Prediction
# --------------------------
def predict_emotion(file):
    model, device = load_model()
    tensor = torch.tensor(process_audio(file), dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        return EMOTION_LABELS[pred_idx]

# --------------------------
# Streamlit UI
# --------------------------
def main():
    st.set_page_config(
        page_title="🎧 Emotion Recognition",
        page_icon="🎤",
        layout="centered"
    )
    
    st.title("🎤 Speech Emotion Recognizer")
    st.markdown("""
    Upload a `.wav` audio file and the model will classify the emotion.
    *Supported emotions:* neutral, calm, happy, sad, angry, fear, disgust, surprise
    """)
    
    with st.expander("💡 How it works"):
        st.markdown("""
        1. The AI model analyzes acoustic features:
           - Zero Crossing Rate
           - Chroma STFT
           - MFCC coefficients
           - RMS energy
           - Mel Spectrogram
        2. Features are standardized and fed to a CNN-LSTM model
        3. Emotion prediction is made from 8 possible classes
        """)
    
    uploaded_file = st.file_uploader(
        "Upload your .wav file", 
        type=["wav"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        st.subheader("Audio Preview")
        st.audio(uploaded_file, format="audio/wav")
        
        with st.spinner("🔍 Analyzing emotion..."):
            try:
                prediction = predict_emotion(uploaded_file)
                st.success(f"🎯 Predicted Emotion: **{prediction.upper()}**")
                
                # Display confidence distribution
                model, device = load_model()
                tensor = torch.tensor(process_audio(uploaded_file), 
                                     dtype=torch.float32).to(device)
                with torch.no_grad():
                    output = model(tensor)
                    probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                
                st.subheader("Confidence Distribution")
                prob_df = pd.DataFrame({
                    "Emotion": EMOTION_LABELS,
                    "Confidence": probabilities
                })
                st.bar_chart(prob_df.set_index("Emotion"))
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()
