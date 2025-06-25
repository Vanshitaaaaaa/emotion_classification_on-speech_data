# Speech Emotion Recognition System

## Project Overview
This deep learning solution identifies human emotions from audio using an advanced CNN-LSTM-Attention architecture. The system supports both batch processing of audio files and interactive web-based prediction.

## Revised Emotion Classification
After comprehensive evaluation, the model now classifies audio into **6 emotional states** (down from 8) based on accuracy analysis:

| Emotion   | Accuracy | Status       |
|-----------|----------|--------------|
| **happy**   | 82.67%      | ✅ Retained  |
| **angry**   | 78.67%      | ✅ Retained  |
| **surprise**| 71.79%      | ⚠️ Marginal  |
| **disgust** | 46.15%      | ❌ Dropped |
| **neutral** | 76.32%      | ✅ Retained |
| **calm**    | 88.00%      | ✅ Retained  |
| **sad**     | 73.33%      | ⚠️ Marginal   |
| **fearful** | 53.33%     | ❌ Dropped   |

### Accuracy Improvements
- **Original 8-class accuracy**: 72.71%
- **Revised 6-class accuracy**: 79.31%
- **Revised 4-class accuracy**: 82.13%
- **Macro F1 Score**: 81.77%

## Technical Implementation
### Audio Processing Pipeline
1. **Audio Loading**  
   - 2.5s duration with 0.6s offset
2. **Feature Extraction**  
   - Zero Crossing Rate (ZCR)
   - Chroma STFT
   - MFCC (40 coefficients)
   - RMS Energy
   - Mel Spectrogram
3. **Feature Aggregation**  
   - Temporal mean pooling
4. **Standardization**  
   - Scikit-learn StandardScaler
5. **Input Preparation**  
   - Reshaped for CNN compatibility

### Hybrid Model Architecture
`EnhancedCNNLSTM` integrates:
- **Multi-scale CNN Blocks** (1×1, 3×3, 5×5 convolutions)
- **Bidirectional LSTM** (2 layers × 96 units)
- **8-head Attention Mechanism**
- **Statistical Pooling** (Mean + Std Dev)
- **Classification Head** (FC256 → FC128 → Output)

## Performance Comparison
| Model Type          | Accuracy | Dataset     |
|---------------------|----------|-------------|
| Baseline (8-class)  | 72.71%    | RAVDESS     |
| **Revised (6-class)** | **79.31%** | RAVDESS     |
| **Revised (4-class)** | **82.13%** | RAVDESS     |
| Vision Transformer  | 71.6%    | RAVDESS [2] |
| Conv1D CNN          | 92%      | RAVDESS [3] |
| LSTM+CNN            | 97.5%    | Custom [5]  |

## Implementation Guide

### 1. Clone Repository
```bash
git clone https://github.com/Vanshitaaaaaa/emotion_classification_on-speech_data
cd emotion_classification_on-speech_data

```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Batch Processing
```bash
python testing.py
```
**Input**: Folder containing `.wav` files  
**Output**: Terminal predictions + `emotion_predictions.csv`

### 4. Launch Web App
```bash
streamlit run app.py
```

## Future Enhancements
1. **Ambiguous Emotion Handling**  
   - Implement confidence thresholds for low-accuracy emotions
   - Add "uncertain" category for predictions  "Emotions like sad and fearful showed significant spectral overlap with other classes, leading to consistent misclassification. Dropping these classes improved overall system reliability." - Model Evaluation Report

