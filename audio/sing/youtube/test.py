from pytube import YouTube

def download_audio_segment(ytid, output_path='path_to_save_audio'):
    try:
        # Create a YouTube object
        youtube_url = f'https://www.youtube.com/watch?v={ytid}'
        yt = YouTube(youtube_url)
        
        print(youtube_url)

        # Get the highest resolution audio stream
        audio_stream = yt.streams.filter(only_audio=True).first()

        # Download the audio segment based on start and end times
        output_filename = f'{ytid}.mp3'
        output_filepath = f'{output_path}/{output_filename}'
        audio_stream.download(output_path=output_path, filename=output_filename)

        print(f"Audio segment download successful: {output_filepath}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
video_data = "RqS924W5td0, 180.000, 190.000, '/m/04rlf,/m/07qb_dv'"
ytid, start_seconds, end_seconds, positive_label = video_data.split(', ')

print(ytid)

download_audio_segment(ytid,output_path='audio/sing/youtube')

