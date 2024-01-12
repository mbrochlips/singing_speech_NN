import os
import torch
from tqdm.notebook import trange
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np
import librosa
from scipy.signal import stft
from tqdm.notebook import tqdm, trange
from scipy.stats import binom
import shutil

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

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

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
def convert_sound(filename, type, save_dir):
    # print(1)
    # Load sound from file
    sample_rate, sound = read_mp3(filename)
    audio_length_seconds = len(sound) / sample_rate  # Calculate the audio length in seconds
    # Compute spectrogram
    _, _, Z = stft(sound, fs=sample_rate, nperseg=446, noverlap=400)
    # Log of absolute value, scaled between 0 and 1
    Z = np.clip(np.log(np.abs(Z))/10+1, 0, 1)
    # Split spectrogram into a sequence of "grey-scale images"
    window_length = 224
    step_size = 100 if type == "train" else 224 # STEP SIZE FOR TRAIN DATA
    num_windows = (Z.shape[1]-window_length)//step_size + 1
    spectrograms = np.array([Z[:, (i*step_size):(i*step_size+window_length)] for i in range(num_windows)])
    time_per_spectrogram = audio_length_seconds/spectrograms.shape[0]
    # Expand "color" axis
    spectrograms = np.repeat(spectrograms[:,None], 3, axis=1)
    # Apply appropriate preprocessing from neural network
    #T = preprocess(torch.tensor(spectrograms))
    spectrogram_files = []
    for i, spec in enumerate(spectrograms):
        spec_file = os.path.join(save_dir, f"{os.path.basename(filename)}_spec_{i}.npy")
        np.save(spec_file, spec)
        spectrogram_files.append(spec_file)
    return spectrogram_files, time_per_spectrogram

# Function to list MP3 files in a directory
def list_mp3_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp3')]

class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        spec = np.load(self.file_paths[idx])
        if self.transform:
            spec = self.transform(torch.tensor(spec))
        label = self.labels[idx]
        return spec, label

def create_dataloader(speech_files, singing_files, type, save_dir, transform):
    all_file_paths = []
    labels = []
    totaltimes = 0
    
    for f in speech_files + singing_files:
        spec_files, time_per_spec = convert_sound(f, type, save_dir)
        totaltimes += time_per_spec
        all_file_paths.extend(spec_files)
        labels.extend([0 if f in speech_files else 1] * len(spec_files))
    
    dataset = SpectrogramDataset(all_file_paths, labels, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True), totaltimes / len(all_file_paths)

# Define the save directories for training and testing spectrograms
save_dir_train = "C:/Users/oscar/Downloads/audioOptimized/spectrograms_train"
save_dir_test = "C:/Users/oscar/Downloads/audioOptimized/spectrograms_test"

# Clear spectrogram directories before running the code
clear_directory(save_dir_train)
clear_directory(save_dir_test)

# Paths to speech and singing folders
speech_train_folder = "C:/Users/oscar/Downloads/audioOptimized/train/speech"
singing_train_folder = "C:/Users/oscar/Downloads/audioOptimized/train/sing"

# Load and prepare training data
speech_train_files = list_mp3_files(speech_train_folder)
singing_train_files = list_mp3_files(singing_train_folder)
train_data, avg_time_train = create_dataloader(speech_train_files, singing_train_files, "train", save_dir_train, preprocess)

###################
# Optimizer that only updates the parameters of the classifier
optimizer = torch.optim.Adam(model.classifier[1].parameters(), lr=0.01)

# Loss function 
loss_function = torch.nn.BCELoss(reduction='sum')

# Run training loop
epochs = 15
model.train()
with trange(epochs) as epoch_range:
    for epoch in epoch_range:
        training_loss = 0
        for X, y in train_data:
            model.zero_grad()
            y_estimate = model(X).squeeze()
            y = y.float()
            loss = loss_function(y_estimate, y)
            loss.backward()
            optimizer.step()
            training_loss += loss.detach().numpy()
        print(f'Epoc: {epoch}, Training loss: {training_loss}')



#######################
# Load and prepare test data
print("Testing")
speech_test_folder = "C:/Users/oscar/Downloads/audioOptimized/test/speech"
singing_test_folder = "C:/Users/oscar/Downloads/audioOptimized/test/sing"



###################
# AUDIO FILE TEST
speech_test_files = list_mp3_files(speech_test_folder)
singing_test_files = list_mp3_files(singing_test_folder)
test_data, avg_time_test = create_dataloader(speech_test_files, singing_test_files, "test", save_dir_test, preprocess)

# Test loop (Model evaluation - file by file)
model.eval()
total_files = correct_files = 0
speech_classification_results = []  # Array for results from the speech test folder
singing_classification_results = []  # Array for results from the sing test folder

for file in speech_test_files + singing_test_files:
    # Convert each file to spectrogram files and get the total number of spectrograms
    spectrogram_files, _ = convert_sound(file, "test", save_dir_test)  # This saves spectrograms to disk
    file_label = 0 if file in speech_test_files else 1  # Label: 0 for speech, 1 for singing

    # Initialize list to store predictions for the file
    predictions = []

    # Process each saved spectrogram file
    for spec_file in spectrogram_files:
        # Load the spectrogram from disk
        spectrogram = np.load(spec_file)
        spectrogram_tensor = preprocess(torch.tensor(spectrogram).unsqueeze(0))  # Add batch and color channel dimensions
        y_estimate = model(spectrogram_tensor).squeeze()  # Remove batch dimension from output
        predictions.append(y_estimate.item())

    # Calculate median prediction for the file
    median_prediction = np.median(predictions)
    # Determine classification based on the median prediction
    file_classification = 1 if median_prediction >= 0.5 else 0

    # Check if the classification is correct and increment counters
    total_files += 1
    if file_classification == file_label:
        correct_files += 1

    # Append result to the corresponding array
    classification_str = "Speech" if file_classification == 0 else "Singing"
    if file_label == 0:  # Speech
        speech_classification_results.append(f"{file[-11:]}: {classification_str}")
    else:  # Singing
        singing_classification_results.append(f"{file[-11:]}: {classification_str}")

# Calculate accuracy
accuracy = correct_files / total_files

# Print the results arrays
print("Speech Test Folder Results:", speech_classification_results)
print("Singing Test Folder Results:", singing_classification_results)

# Calculate 95% confidence interval for the accuracy
z = 1.96  # z-score for 95% confidence
p = accuracy  # proportion of successes
interval_lower = (p + z**2/(2*total_files) - z*np.sqrt(p*(1-p)/total_files + z**2/(4*total_files**2))) / (1 + z**2/total_files)
interval_upper = (p + z**2/(2*total_files) + z*np.sqrt(p*(1-p)/total_files + z**2/(4*total_files**2))) / (1 + z**2/total_files)

# Print accuracy and confidence interval
print(f'Accuracy: {accuracy*100:0.2f}%')
print(f'95% Confidence Interval: [{interval_lower*100:.2f}%, {interval_upper*100:.2f}%]')



####################
# SPECTROGRAM TEST
speech_test_files = list_mp3_files(speech_test_folder)
singing_test_files = list_mp3_files(singing_test_folder)
test_data, _ = create_dataloader(speech_test_files, singing_test_files, "test", save_dir_test, preprocess)

# Test loop (Model evaluation)
# Evaluate model performance
model.eval()
total = correct = 0
for X, y in test_data:
    y_estimate = model(X)
    # Ensure y is a float tensor and has the same shape as y_estimate
    y = y.float().view_as(y_estimate)
    # Perform the comparison and sum up correctly predicted cases
    correct += (y_estimate.round() == y).sum().item()
    total += y.size(0)


# Calculate 95% confidence interval for the accuracy
accuracy = correct / total
z = 1.96  # z-score for 95% confidence
n = total  # total number of samples
p = accuracy  # proportion of successes
interval_lower = (p + z**2/(2*n) - z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)
interval_upper = (p + z**2/(2*n) + z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / (1 + z**2/n)

# Print accuracy and confidence interval
print(f'Accuracy: {accuracy*100:0.2f}%')
print(f'95% Confidence Interval: [{interval_lower*100:.2f}%, {interval_upper*100:.2f}%]')
