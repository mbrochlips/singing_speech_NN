# Import necessary libraries
import os
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np
import librosa
from scipy.signal import stft
from tqdm.notebook import tqdm, trange

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
def read_mp3(filename, as_float=True, duration = 2.0): # Change duration here
    sound, sample_rate = librosa.load(filename, sr=None, mono=True, duration= duration)
    if as_float:
        sound = sound.astype(float)
    return sample_rate, sound

# Convert sound to spectrogram "images"
def convert_sound(filename):
    print(1)
    # Load sound from fileF
    sample_rate, sound = read_mp3(filename)
    # Compute spectrogram
    time, frequency, Z = stft(sound, fs=sample_rate, nperseg=446, noverlap=400)
    # Log of absolute value, scaled between 0 and 1
    Z = np.clip(np.log(np.abs(Z))/10+1, 0, 1)
    # Split spectrogram into a sequence of "grey-scale images"
    window_length = 224
    step_size = 100
    num_windows = (Z.shape[1]-window_length)//step_size + 1
    spectrograms = np.array([Z[:, (i*step_size):(i*step_size+window_length)] for i in range(num_windows)])
    # Expand "color" axis
    spectrograms = np.repeat(spectrograms[:,None], 3, axis=1)
    # Apply appropriate preprocessing from neural network
    T = preprocess(torch.tensor(spectrograms))
    return T

# Function to list MP3 files in a directory
def list_mp3_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp3')]

# Modified create_dataloader function to handle speech and singing data
def create_dataloader(speech_files, singing_files):
    print(2)
    # Process speech and singing files
    speech_data = [convert_sound(f) for f in speech_files]
    singing_data = [convert_sound(f) for f in singing_files]

    # Combine and label data
    all_data = speech_data + singing_data
    labels = [0] * len(speech_data) + [1] * len(singing_data)
    y = torch.tensor(np.hstack([[label]*len(data) for label, data in zip(labels, all_data)]))[:,None].float()

    # Stack and create dataset
    X = torch.vstack(all_data)
    dataset = torch.utils.data.TensorDataset(X, y)

    # Return DataLoader
    return torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

# Paths to speech and singing folders
speech_folder = 'C:/Users/oscar/Downloads/Testclips/Speech'
singing_folder = 'C:/Users/oscar/Downloads/Testclips/Singing'

# Load and prepare training data
speech_train_files = list_mp3_files(speech_folder)
singing_train_files = list_mp3_files(singing_folder)
train_data = create_dataloader(speech_train_files, singing_train_files)

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
speech_test_folder = 'C:/Users/oscar/Downloads/Testclips/Speech_test'
singing_test_folder = 'C:/Users/oscar/Downloads/Testclips/Singing_test'

speech_test_files = list_mp3_files(speech_test_folder)
singing_test_files = list_mp3_files(singing_test_folder)
test_data = create_dataloader(speech_test_files, singing_test_files)

# Test loop (Model evaluation)
# Evaluate model performance
model.eval()
total = correct = 0
for X, y in test_data:
    y_estimate = model(X)
    correct += sum(y_estimate.round()==y).item()
    total += len(y)
print(f'Accuracy: {correct/total*100:0.2f}%')
