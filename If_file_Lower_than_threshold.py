import librosa
import os
# read_mp3 som vi kender den MEN uden duration og den læser HELE FILEN
def read_mp3(filename, as_float=True): # Change duration here
    sound, sample_rate = librosa.load(filename, sr=None, mono=True) # Offset = 1.0 betyder, at lydfilen læses fra 1.0 fra start og 2 sekunder frem (duration = 2.0)
    if as_float:
        sound = sound.astype(float)
    return sample_rate, sound

# Kode som tager path af mappe, spytter ud en liste med filer der er under en threshold
def is_lower_than_threshold(path , threshold:float):
    filenames = []
    for file in os.listdir(path):
        sample_rate , wav = read_mp3(path+file)
        length_in_seconds = len(wav) / sample_rate
        if length_in_seconds < threshold:
            filenames.append(file)
    return filenames



path = '/Users/davidlindahl/Desktop/GitHub/singing_speech_NN-main/audio/train/sing/'
is_lower_than_threshold(path,5.0)