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
    # Load sound from file
    sample_rate, sound = read_mp3(filename)
    audio_length_seconds = len(sound) / sample_rate  # Calculate the audio length in seconds
    # Compute spectrogram
    t, frequency, Z = stft(sound, fs=sample_rate, nperseg=446, noverlap=400)
    # Log of absolute value, scaled between 0 and 1
    Z = np.clip(np.log(np.abs(Z))/10+1, 0, 1)
    # Split spectrogram into a sequence of "grey-scale images"
    window_length = 224
    step_size = 100 if type == "train" else 400 # STEP SIZE FOR TRAIN DATA
    num_windows = (Z.shape[1]-window_length)//step_size + 1
    spectrograms = np.array([Z[:, (i*step_size):(i*step_size+window_length)] for i in range(num_windows)])
    time_per_spectrogram = audio_length_seconds/spectrograms.shape[0]
    # Expand "color" axis
    spectrograms = np.repeat(spectrograms[:,None], 3, axis=1)
    # Apply appropriate preprocessing from neural network
    T = preprocess(torch.tensor(spectrograms))
    return T, time_per_spectrogram

# Function to list MP3 files in a directory
def list_mp3_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp3')]

# Modified create_dataloader function to handle speech and singing data
def create_dataloader(speech_files, singing_files, type):
    all_data = []
    labels = []
    totaltimes = 0
    
    for f in speech_files + singing_files:
        data, time_per_spec = convert_sound(f, type)
        totaltimes += time_per_spec
        all_data.append(data)
        labels.append(0 if f in speech_files else 1)
    
    avg_timeper_sepctrogram = totaltimes / (len(speech_files) + len(singing_files))

    y = torch.tensor(np.hstack([[label] * len(data) for label, data in zip(labels, all_data)]))[:, None].float()
    X = torch.vstack(all_data)
    dataset = torch.utils.data.TensorDataset(X, y)

    return torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True), avg_timeper_sepctrogram

# Paths to speech and singing folders
speech_train_folder = "C:/Users/oscar/Downloads/audiobig/train/speech"
singing_train_folder = "C:/Users/oscar/Downloads/audiobig/train/sing"
# speech_train_folder = os.path.join('audio','train','speech')
# singing_train_folder = os.path.join('audio','train','sing')

# Load and prepare training data
speech_train_files = list_mp3_files(speech_train_folder)
singing_train_files = list_mp3_files(singing_train_folder)
train_data, avg_time_train = create_dataloader(speech_train_files, singing_train_files, type = "train")

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
speech_test_folder = "C:/Users/oscar/Downloads/audiobig/test/speech"
singing_test_folder = "C:/Users/oscar/Downloads/audiobig/test/sing"
# speech_test_folder = os.path.join('audio','test','speech')
# singing_test_folder = os.path.join('audio','test','sing')

speech_test_files = list_mp3_files(speech_test_folder)
singing_test_files = list_mp3_files(singing_test_folder)
test_data, avg_time_test = create_dataloader(speech_test_files, singing_test_files, type = "test")
avg_time_per_spec = (avg_time_train + avg_time_test) / 2

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

# Calculate 95% confidence interval for the accuracy
accuracy = correct / total
z = 1.96  # z-score for 95% confidence
n = total  # total number of samples
p = accuracy  # proportion of successes
interval_lower = (p + z**2/(2*n) - z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)
interval_upper = (p + z**2/(2*n) + z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)

# Calculate batch times
median_batch_test_time = np.median(test_batch_times)
avg_batch_time = avg_time_per_spec * 10
test_time_per_sec = median_batch_test_time / avg_batch_time
print(f'It takes {test_time_per_sec:.4f} s to test on 1 sec of data')

# Print accuracy and confidence interval
print(f'Accuracy: {accuracy*100:0.2f}%')
print(f'95% Confidence Interval: [{interval_lower*100:.2f}%, {interval_upper*100:.2f}%]')
