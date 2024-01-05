# Import necessary libraries
import os
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np
import librosa
import time
from scipy.signal import stft
from tqdm.notebook import tqdm, trange
#
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
def read_mp3(filename, as_float=True, duration = 5.0): # Change duration here
    sound, sample_rate = librosa.load(filename, sr=None, mono=True, duration= duration, offset = 1.0) # Offset = 1.0 betyder, at lydfilen læses fra 1.0 fra start og 2 sekunder frem (duration = 2.0)
    if as_float:
        sound = sound.astype(float)
    return sample_rate, sound


# # Function to read MP3 file using librosa
# def read_mp3(filename, as_float=True): # Change duration here
#     sound, sample_rate = librosa.load(filename, sr=None, mono=True) # Offset = 1.0 betyder, at lydfilen læses fra 1.0 fra start og 2 sekunder frem (duration = 2.0)
#     if as_float:
#         sound = sound.astype(float)
#     return sample_rate, sound


# Convert sound to spectrogram "images"
def convert_sound(filename):
    #print(1)
    start_time = time.time()
    # Load sound from fileF
    sample_rate, sound = read_mp3(filename)
    # Compute spectrogram
    t, frequency, Z = stft(sound, fs=sample_rate, nperseg=446, noverlap=400)
    # Log of absolute value, scaled between 0 and 1
    epsilon = 1e-5
    Z = np.clip(np.log(np.abs(Z) + epsilon) / 10 + 1, 0, 1)
    # Split spectrogram into a sequence of "grey-scale images"
    window_length = 224
    step_size = 100
    num_windows = (Z.shape[1]-window_length)//step_size + 1
    spectrograms = np.array([Z[:, (i*step_size):(i*step_size+window_length)] for i in range(1,num_windows)])
    # Expand "color" axis
    spectrograms = np.repeat(spectrograms[:,None], 3, axis=1)
    # Apply appropriate preprocessing from neural network
    T = preprocess(torch.tensor(spectrograms))
    end_time = time.time()
    # print(f'Runtime for making spectrograms: {end_time - start_time:4f} seconds')
    return T, end_time - start_time  # Return the data and processing time

# Function to list MP3 files in a directory
def list_mp3_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp3')]

# Modified create_dataloader function to handle speech and singing data
def create_dataloader(speech_files, singing_files):
    spectrogram_times = []  # List to store spectrogram processing times
    all_data = []
    labels = []
    
    for f in speech_files + singing_files:
        data, processing_time = convert_sound(f)
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
speech_folder = '/Users/davidlindahl/Desktop/GitHub/singing_speech_NN-main/audio/train/speech'
singing_folder = '/Users/davidlindahl/Desktop/GitHub/singing_speech_NN-main/audio/train/sing'

# Load and prepare training data
speech_train_files = list_mp3_files(speech_folder)
singing_train_files = list_mp3_files(singing_folder)
train_data, train_spectrogram_times = create_dataloader(speech_train_files, singing_train_files)

###################
# Optimizer that only updates the parameters of the classifier
optimizer = torch.optim.Adam(model.classifier[1].parameters(), lr=0.01)

# Loss function 
loss_function = torch.nn.BCELoss(reduction='sum')

# Run training loop
epochs = 5
model.train()
with trange(epochs) as epoch_range:
    for epoch in epoch_range:
        print(3)
        training_loss = 0
        for X, y in train_data:
            model.zero_grad()
            y_estimate = model(X)
            loss = loss_function(y_estimate, y)
            loss.backward()
            optimizer.step()
            training_loss += loss.detach().numpy()
        epoch_range.set_description_str(f'Training loss: {training_loss:.1f}, Progress')

#######################
# Load and prepare test data
print("Testing")
speech_test_folder = '/Users/davidlindahl/Desktop/GitHub/singing_speech_NN-main/audio/test/speech'
singing_test_folder = '/Users/davidlindahl/Desktop/GitHub/singing_speech_NN-main/audio/test/sing'

speech_test_files = list_mp3_files(speech_test_folder)
singing_test_files = list_mp3_files(singing_test_folder)
test_data, test_spectrogram_times = create_dataloader(speech_test_files, singing_test_files)

# Test loop (Model evaluation)
# Evaluate model performance
model.eval()
total = correct = 0
test_batch_times = []  # List to store test batch processing times
for X, y in test_data:
    start_test = time.time()
    y_estimate = model(X)
    end_test = time.time()
    test_batch_times.append(end_test - start_test)  # Accumulate test batch time
    correct += sum(y_estimate.round() == y).item()
    total += len(y)


print_stats_and_outliers(train_spectrogram_times + test_spectrogram_times, "spectrogram processing")
print_stats_and_outliers(test_batch_times, "test batch")
print(f'Accuracy: {correct/total*100:0.2f}%')


##############
# Accuracy and time log (ON POWERFUL COMPUTER THOUGH. Check on worse specs)
##############
# Test time is for 0,1s
##0.01s
# 84,85% acc
# spect time: 0,007s
# test time: 0,042s

##0.1s
# 87,88%
# spect time: 0,007s
# test time: 0,040s

##0.25s
# 90,91% acc
# spect time: 0,007s
# test time: 0,041s

##0.5s
# 85,5% acc
# spect time: 0,007s
# test time: 0,035s

## 1s:
# 85,3% acc
# spect time: 0,016s (tons of smaller outliers)
# test time: 0,046s

## 2s:
#93,2% acc
# spect time: 0,037s
# test time: 0,046s

## 3s
# 90,54% acc
# spect time: 0,043s
# test time: 0,046s

## 4s
# 89,50% acc
# spect time: 0,057s
# test time: 0,045s

#uncapped
# 95%
# spect time: 0,10s
# test time: 0,43s