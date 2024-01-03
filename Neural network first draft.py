import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.model_selection import train_test_split

# Directory paths for the audio files
# speech_dir = "C:\Users\oscar\Downloads\Files"
# singing_dir = "C:\Users\oscar\Downloads\Files"

# Function to load an audio file and compute its spectrogram
def compute_spectrogram(file_path):
    # Load audio file
    audio, sr = librosa.load(file_path, sr=None)
    
    # Compute the spectrogram
    S = librosa.feature.melspectrogram(audio, sr=sr, n_fft=2048, hop_length=1024, n_mels=224)
    S_DB = librosa.power_to_db(S, ref=np.max)
    
    # Resize to 224x224 to match MobileNetV2 input
    S_DB_resized = librosa.util.fix_length(S_DB, size=224, axis=1)
    return S_DB_resized

# Function to prepare dataset
def prepare_dataset(speech_dir, singing_dir):
    labels = []
    features = []

    # Process all speech files
    for file in os.listdir(speech_dir):
        file_path = os.path.join(speech_dir, file)
        spec = compute_spectrogram(file_path)
        features.append(spec)
        labels.append(0)  # 0 for speech

    # Process all singing files
    for file in os.listdir(singing_dir):
        file_path = os.path.join(singing_dir, file)
        spec = compute_spectrogram(file_path)
        features.append(spec)
        labels.append(1)  # 1 for singing

    return np.array(features), np.array(labels)

# Load and prepare the dataset
X, y = prepare_dataset(speech_dir, singing_dir)
X = preprocess_input(X[..., np.newaxis])  # Add channel dimension and preprocess

# Split into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
