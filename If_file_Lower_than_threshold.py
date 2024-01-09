import librosa
import os
# read_mp3 som vi kender den MEN uden duration og den læser HELE FILEN
def read_mp3(filename, as_float=True): # Change duration here
    sound, sample_rate = librosa.load(filename, sr=None, mono=True) # Offset = 1.0 betyder, at lydfilen læses fra 1.0 fra start og 2 sekunder frem (duration = 2.0)
    if as_float:
        sound = sound.astype(float)
    return sample_rate, sound


def is_lower_than_threshold(start_path , threshold:float, destination):
    '''Kode som tager path af mappe, spytter ud en liste med filer der er under en threshold.
    path: The path that you want to move files from
    threshold: seconds. If a file is lower than the threshold, they will me moved
    target: target directory.'''
    for file in os.listdir(start_path):
        file_path = os.path.join(start_path,file)
        sample_rate , wav = read_mp3(start_path+file)
        length_in_seconds = len(wav) / sample_rate
        if length_in_seconds < threshold:
            destination_file_path_with_file = os.path.join(destination,file)
            os.rename(file_path,destination_file_path_with_file)


start_path = '/Users/davidlindahl/Desktop/GitHub/singing_speech_NN-main/audio/test/speech/'
destination = '/Users/davidlindahl/Desktop/GitHub/singing_speech_NN-main/audio/train/speech'
is_lower_than_threshold(start_path,5.0,destination)