import librosa
def read_mp3(filename, as_float=True, duration = 2.0): # Duration has been added
    sound, sample_rate = librosa.load(filename, sr=None, mono=True, duration = duration)
    if as_float:
        sound = sound.astype(float)
    return sample_rate, sound

# Tests
read_mp3('audio/sing/00 - audio - q1.mp3')
#KDASKKDSAK