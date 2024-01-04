# Import libraries
import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
from scipy.signal import stft
from tqdm.notebook import tqdm, trange

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


# Reads sound (wav) file and computes windowed spectrograms 
# as "images" of size 224 x 224
def convert_sound(filename):
    # Load sound from file
    sample_rate, sound = scipy.io.wavfile.read(filename)
    # Convert to floating point values between -1 and 1
    sound = sound.astype(float) / np.max(np.abs(sound)) 
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


train1 = "C:/Users/oscar/Downloads/Tests2.wav"
# train2 = "C:/Users/oscar/Downloads/Speech/common_voice_en_38487418.mp3"

# Show an example of a 224 x 224 specrogram "image"
T = convert_sound(train1)
image_number = 0
color_channel = 0
plt.imshow(T[image_number, color_channel], origin='lower', cmap='afmhot_r')
plt.show()