import os
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from model import EnhancedCNNLSTM

# ====================
# Feature Extraction
# ====================
def extract_features(data, sample_rate):
    """Extract audio features with improved efficiency"""
    features = []
    
    # Zero Crossing Rate
    features.append(np.mean(librosa.feature.zero_crossing_rate(y=data)))
    
    # Chroma STFT
    stft = np.abs(librosa.stft(data))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    features.extend(np.mean(chroma, axis=1))
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))
    
    # RMS Energy
    features.append(np.mean(librosa.feature.rms(y=data)))
    
    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=data, sr=sample_rate)
    features.append(np.mean(mel))
    
    return np.array(features)

def get_features(path):
    """Load audio and extract features with error handling"""
    try:
        data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
        return extract_features(data, sample_rate)
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(path)}: {str(e)}")
        return None

# ====================
# Inference
# ====================
def predict_emotions(folder_path):
    """Predict emotions for all WAV files in a folder"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    
    # Load model with error handling
    try:
        model = EnhancedCNNLSTM(num_classes=8)
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return

    emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
    scaler = StandardScaler()

    # Collect valid files and features
    valid_files = []
    features_list = []
    
    print("\n🔍 Scanning audio files...")
    audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
    
    for file in tqdm(audio_files, desc="Processing audio"):
        path = os.path.join(folder_path, file)
        features = get_features(path)
        if features is not None:
            features_list.append(features)
            valid_files.append(file)
    
    if not valid_files:
        print("❌ No valid WAV files found")
        return
    
    # Process features
    features_array = np.array(features_list)
    features_scaled = scaler.fit_transform(features_array)
    
    # Make predictions
    results = []
    print("\n🎯 Making predictions...")
    
    for i in tqdm(range(len(features_scaled)), desc="Classifying emotions"):
        sample = features_scaled[i].reshape(1, 1, 9, 18)
        tensor = torch.tensor(sample, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            output = model(tensor)
            pred_idx = torch.argmax(output, dim=1).item()
            emotion = emotion_labels[pred_idx]
            results.append((valid_files[i], emotion))
    
    # Save and display results
    df = pd.DataFrame(results, columns=["filename", "predicted_emotion"])
    df.to_csv("emotion_predictions.csv", index=False)
    
    print("\n📊 Prediction Results:")
    print(df.to_string(index=False))
    print(f"\n✅ Saved predictions to emotion_predictions.csv")

# ====================
# Entry Point
# ====================
if __name__ == "__main__":
    folder = input("\n📁 Enter path to folder containing .wav files: ").strip()
    
    if os.path.isdir(folder):
        predict_emotions(folder)
    else:
        print("❌ Error: Invalid folder path. Please enter a valid directory.")
