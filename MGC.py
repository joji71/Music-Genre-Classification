from flask import Flask, request, render_template
import pickle
import librosa
import numpy as np
import pandas as pd
import csv
import warnings
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load pre-trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('StandardScaler.pkl', 'rb'))

def extract_metadata(filename):
    """
    Extracts audio features from a WAV file and returns them as a DataFrame.
    """
    y, sr = librosa.load(filename, mono=True, duration=30)

    # Compute various audio features
    features = {
        "chroma_stft": np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        "spectral_centroid": np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        "rolloff": np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        "zero_crossing_rate": np.mean(librosa.feature.zero_crossing_rate(y=y))
    }

    # Compute MFCC features (13 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc{i+1}'] = np.mean(mfccs[i])

    # Convert dictionary to DataFrame
    df = pd.DataFrame([features])
    return df

@app.route("/")
def home():
    """
    Renders the homepage template.
    """
    return render_template('homepage.html')

@app.route("/success", methods=['POST'])
def success():
    """
    Handles file upload, extracts metadata, predicts genre, and renders result.
    """
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)  # Save uploaded file locally
        
        # Validate file type
        if not f.filename.endswith('.wav'):
            return render_template('genre.html', name="Invalid file type!", genre="Please upload only WAV files.")

        # Extract metadata and make prediction
        metadata = extract_metadata(f.filename)
        scaled_metadata = scaler.transform(metadata)
        predicted_genre = model.predict(scaled_metadata)[0]

        # Genre mapping
        genre_map = {0: 'Blues', 1: 'Classical', 2: 'Country', 3: 'Jazz', 4: 'Metal', 5: 'Pop'}
        genre_name = genre_map.get(predicted_genre, "Unknown")

        return render_template('genre.html', name=f.filename, genre=genre_name)

if __name__ == '__main__':
    app.run()
