# Import libraries
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from tqdm.notebook import tqdm, trange
import librosa

# Set up neural network
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
def read_mp3(filename, as_float=True):
    sound, sample_rate = librosa.load(filename, sr=None, mono=True)
    if as_float:
        sound = sound.astype(float)
    return sample_rate, sound

# Reads sound (MP3) file and computes windowed spectrograms 
# as "images" of size 224 x 224
def convert_sound(filename):
    # Load sound from file
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


train1 = 'C:/Users/oscar/Downloads/Speech/common_voice_en_38487417.mp3'
train2 = "C:/Users/oscar/Downloads/Speech/common_voice_en_38487418.mp3"

# test1 = "C:/Users/oscar/Downloads/Speech/common_voice_en_38487419.mp3"
# test2 = "C:/Users/oscar/Downloads/Speech/common_voice_en_38487420.mp3"

# Show an example of a 224 x 224 spectrogram "image"
# T = convert_sound(train1)
# print(T.shape)

# # Show an example of a 224 x 224 specrogram "image"
# T = convert_sound(train1)
# image_number = 1
# color_channel = 0
# plt.imshow(T[image_number, color_channel], origin='lower', cmap='afmhot_r')
# plt.show()



# Create a dataloader from list of wav files
def create_dataloader(filenames):
    # Load and convert the files
    T = [convert_sound(f) for f in filenames]
    # Stack files and labels in torch tensors
    X = torch.vstack(T)
    y = torch.tensor(np.hstack([[i]*len(t) for i, t in enumerate(T)]))[:,None].float()
    # Put data into a dataset object 
    dataset = torch.utils.data.TensorDataset(X, y)
    # Return a dataloader object
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    return dataloader



# Load training data
train_files = [train1, train2]
train_data = create_dataloader(train_files)

# Optimizer that only updates the parameters of the classifier
optimizer = torch.optim.Adam(model.classifier[1].parameters(), lr=0.01)

# Loss function 
loss_function = torch.nn.BCELoss(reduction='sum')

# Run training loop
epochs = 5
model.train()
with trange(epochs) as epoch_range:
    for epoch in epoch_range:
        training_loss = 0
        for X, y in train_data:
            model.zero_grad()
            y_estimate = model(X)
            loss = loss_function(y_estimate, y)
            loss.backward()
            optimizer.step()
            training_loss += loss.detach().numpy()
        epoch_range.set_description_str(f'Training loss: {training_loss:.1f}, Progress')


# Load test data
test_files = [test1, test2]
test_data = create_dataloader(test_files)

# Evaluate model performance
model.eval()
total = correct = 0
for X, y in test_data:
    y_estimate = model(X)
    correct += sum(y_estimate.round()==y).item()
    total += len(y)
print(f'Accuracy: {correct/total*100:0.2f}%')


# Load training data
train_files = ['music1_train.wav', 'music2_train.wav']
train_data = create_dataloader(train_files)

# Optimizer that only updates the parameters of the classifier
optimizer = torch.optim.Adam(model.classifier[1].parameters(), lr=0.01)

# Loss function 
loss_function = torch.nn.BCELoss(reduction='sum')

# Run training loop
epochs = 5
model.train()
with trange(epochs) as epoch_range:
    for epoch in epoch_range:
        training_loss = 0
        for X, y in train_data:
            model.zero_grad()
            y_estimate = model(X)
            loss = loss_function(y_estimate, y)
            loss.backward()
            optimizer.step()
            training_loss += loss.detach().numpy()
        epoch_range.set_description_str(f'Training loss: {training_loss:.1f}, Progress')


# Load test data
test_files = ['music1_test.wav', 'music2_test.wav']
test_data = create_dataloader(test_files)

# Evaluate model performance
model.eval()
total = correct = 0
for X, y in test_data:
    y_estimate = model(X)
    correct += sum(y_estimate.round()==y).item()
    total += len(y)
print(f'Accuracy: {correct/total*100:0.2f}%')       