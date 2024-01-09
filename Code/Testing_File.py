import os
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np
import librosa
import time
from scipy.signal import stft
from tqdm.notebook import tqdm, trange
from scipy.stats import binom

# Neural network setup
weights = MobileNet_V2_Weights.DEFAULT
preprocess = weights.transforms()
model = mobilenet_v2(weights=weights).eval()

# Replace the last layer with a binary classifier
for p in model.parameters():
    p.requires_grad = False
model.classifier[1] = torch.nn.Sequential(
    torch.nn.Linear(model.last_channel, 1),
    torch.nn.Sigmoid())

# Function to read MP3 file using librosa
def read_mp3(filename, as_float=True, duration=0.0):  # Default duration set to 0.0
    # If duration is 0, load the entire file, else load the specified duration
    if duration == 0.0:
        sound, sample_rate = librosa.load(filename, sr=None, mono=True)
    else:
        sound, sample_rate = librosa.load(filename, sr=None, mono=True, duration=duration, offset=1.0)

    if as_float:
        sound = sound.astype(float)

    return sample_rate, sound


# Convert sound to spectrogram "images"
def convert_sound(filename, type):
    # print(1)
    start_time = time.time()
    # Load sound from file
    sample_rate, sound = read_mp3(filename)
    # Compute spectrogram
    t, frequency, Z = stft(sound, fs=sample_rate, nperseg=446, noverlap=400)
    # Log of absolute value, scaled between 0 and 1
    Z = np.clip(np.log(np.abs(Z))/10+1, 0, 1)
    # Split spectrogram into a sequence of "grey-scale images"
    if type == "train":
        window_length = 224
        step_size = 100 # Step size for training data
    elif type == "test":
        window_length = 224
        step_size = 400 # Step size for test data
    num_windows = (Z.shape[1]-window_length)//step_size + 1
    spectrograms = np.array([Z[:, (i*step_size):(i*step_size+window_length)] for i in range(num_windows)])
    print("Spectrogram shape:", spectrograms.shape)
    # Expand "color" axis
    spectrograms = np.repeat(spectrograms[:,None], 3, axis=1)
    # Apply appropriate preprocessing from neural network
    T = preprocess(torch.tensor(spectrograms))
    end_time = time.time()
    return T, end_time - start_time  # Return the data and processing time

# Function to list MP3 files in a directory
def list_mp3_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp3')]


# Function to calculate expected number of windows for a given audio file
def calculate_expected_windows(filename, window_length, step_size):
    sample_rate, sound = read_mp3(filename)

    # Calculate the number of columns in the spectrogram
    num_columns = (len(sound) - 446) // (446 - 400) + 1

    # Calculate the number of spectrograms (or windows) per audio file length
    num_windows = (num_columns - window_length) // step_size + 1
    return int(num_windows)

# Modified create_dataloader function to handle speech and singing data
def create_dataloader(speech_files, singing_files, type):
    spectrogram_times = []  # List to store spectrogram processing times
    all_data = []
    labels = []
    
    for f in speech_files + singing_files:
        # Calculate expected windows
        window_length = 224
        step_size = 100 if type == "train" else 400
        expected_windows = calculate_expected_windows(f, window_length, step_size)
        
        # Process sound and get actual windows
        if type == "train":
            data, processing_time = convert_sound(f, "train")
        elif type == "test":
            data, processing_time = convert_sound(f, "test")
        
        # Print expected and actual window counts
        actual_windows = data.shape[0]
        print(f"File: {f}, Expected Windows: {expected_windows}, Actual Windows: {actual_windows}")
        
        all_data.append(data)
        spectrogram_times.append(processing_time)
        labels.append(0 if f in speech_files else 1)

    y = torch.tensor(np.hstack([[label] * len(data) for label, data in zip(labels, all_data)]))[:, None].float()
    X = torch.vstack(all_data)
    dataset = torch.utils.data.TensorDataset(X, y)

    return torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True), spectrogram_times

def print_stats_and_outliers(times, description):
    median_time = np.median(times)
    std_dev = np.std(times)
    outliers = [t for t in times if abs(t - median_time) > 2 * std_dev]

    print(f"Median {description} time: {median_time:.3f}s")
    print(f"Outliers in {description}: {outliers}")

# Paths to speech and singing folders
speech_folder = 'C:/Users/oscar/Downloads/Testclips/Speech_test'
singing_folder = 'C:/Users/oscar/Downloads/Testclips/Empty'

# Load and prepare training data
speech_train_files = list_mp3_files(speech_folder)
singing_train_files = list_mp3_files(singing_folder)
train_data, train_spectrogram_times = create_dataloader(speech_train_files, singing_train_files, type = "test")
