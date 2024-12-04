from pydub import AudioSegment

def trim_audio(input_file, output_file, start_time, end_time):
    """
    Trims an audio file to the specified start and end times.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the trimmed audio file.
        start_time (int): Start time in milliseconds.
        end_time (int): End time in milliseconds.
    """
    # Load the audio file
    audio = AudioSegment.from_file(input_file)

    # Trim the audio
    trimmed_audio = audio[start_time:end_time]

    # Export the trimmed audio
    trimmed_audio.export(output_file, format="wav")
    print(f"Trimmed audio saved to {output_file}")

# Example usage
if __name__ == "__main__":
    input_path = r"C:\Users\OMOLP094\Downloads\Eminem-Lose-Yourself-Instrumental-Prod.-By-Luis-Resto-Jeff-Bass-Eminem.mp3"  # Replace with your input file path
    output_path = r"C:\Users\OMOLP094\Downloads\trimmed_audio.wav"  # Replace with your desired output file path
    start = 0  # Start time in milliseconds (e.g., 5000ms = 5 seconds)
    end = 29000   # End time in milliseconds (e.g., 15000ms = 15 seconds)

    trim_audio(input_path, output_path, start, end)
