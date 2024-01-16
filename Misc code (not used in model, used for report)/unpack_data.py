import os
from pydub import AudioSegment

def unpack(path): 
    
    # Ensure the directory path ends with a path separator
    if not path.endswith(os.path.sep):
        path += os.path.sep

     # Create mp3_files directory if it doesn't exist
    output_dir = os.path.join(path, 'mp3_files')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, filenames in os.walk(path): # Iterates through all files in directory AND all files in root
        for filename in filenames: # Iterate through all of the files
                if filename.endswith('.mp3'):
                    output_path = os.path.join(output_dir, filename) 
                    os.rename(os.path.join(path,root,filename) , os.path.join(output_dir,filename))


                    