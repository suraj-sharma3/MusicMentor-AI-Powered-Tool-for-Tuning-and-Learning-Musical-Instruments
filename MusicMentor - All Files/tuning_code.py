import streamlit as st
import pyaudio
import numpy as np

# Define tuning frequencies for each instrument
TUNING = {
    "Guitar": {
        "E2": 82.41,
        "A2": 110.00,
        "D3": 146.83,
        "G3": 196.00,
        "B3": 246.94,
        "E4": 329.63,
    },
    "Violin": {
        "G3": 196.00,
        "D4": 293.66,
        "A4": 440.00,
        "E5": 659.26,
    },
    "Piano": {
        "A0": 27.50, "A#0": 29.14, "B0": 30.87, "C1": 32.70, "C#1": 34.65, "D1": 36.71, "D#1": 38.89, "E1": 41.20,
        "F1": 43.65, "F#1": 46.25, "G1": 49.00, "G#1": 51.91, "A1": 55.00, "A#1": 58.27, "B1": 61.74, "C2": 65.41,
        "C#2": 69.30, "D2": 73.42, "D#2": 77.78, "E2": 82.41, "F2": 87.31, "F#2": 92.50, "G2": 98.00, "G#2": 103.83,
        "A2": 110.00, "A#2": 116.54, "B2": 123.47, "C3": 130.81, "C#3": 138.59, "D3": 146.83, "D#3": 155.56, "E3": 164.81,
        "F3": 174.61, "F#3": 185.00, "G3": 196.00, "G#3": 207.65, "A3": 220.00, "A#3": 233.08, "B3": 246.94, "C4": 261.63,
        "C#4": 277.18, "D4": 293.66, "D#4": 311.13, "E4": 329.63, "F4": 349.23, "F#4": 369.99, "G4": 392.00, "G#4": 415.30,
        "A4": 440.00, "A#4": 466.16, "B4": 493.88, "C5": 523.25, "C#5": 554.37, "D5": 587.33, "D#5": 622.25, "E5": 659.26,
        "F5": 698.46, "F#5": 739.99, "G5": 783.99, "G#5": 830.61, "A5": 880.00, "A#5": 932.33, "B5": 987.77, "C6": 1046.50,
        "C#6": 1108.73, "D6": 1174.66, "D#6": 1244.51, "E6": 1318.51, "F6": 1396.91, "F#6": 1479.98, "G6": 1567.98, "G#6": 1661.22,
        "A6": 1760.00, "A#6": 1864.66, "B6": 1975.53, "C7": 2093.00, "C#7": 2217.46, "D7": 2349.32, "D#7": 2489.02, "E7": 2637.02,
        "F7": 2793.83, "F#7": 2959.96, "G7": 3135.96, "G#7": 3322.44, "A7": 3520.00, "A#7": 3729.31, "B7": 3951.07, "C8": 4186.01,
    },
}

# Function to detect frequency
def detect_frequency(data, rate):
    fft_data = np.fft.rfft(data)
    frequencies = np.fft.rfftfreq(len(data), d=1.0 / rate)
    magnitude = np.abs(fft_data)
    peak_index = np.argmax(magnitude)
    return frequencies[peak_index]

# Function to find closest note
def find_closest_string(frequency, tuning):
    closest_string = min(tuning, key=lambda key: abs(tuning[key] - frequency))
    return closest_string, tuning[closest_string]

# Function to provide tuning feedback
def provide_tuning_feedback(detected_frequency, closest_string, expected_frequency):
    difference = detected_frequency - expected_frequency
    if abs(difference) < 1:
        return f"{closest_string} is in tune! 🎵"
    elif difference > 0:
        return f"{closest_string} is sharp (too high). Loosen the string slightly."
    else:
        return f"{closest_string} is flat (too low). Tighten the string slightly."

# Streamlit UI
st.title("Instrument Tuning Assistant")
st.write("Select the instrument you want to tune and follow the instructions.")

instrument = st.selectbox("Choose an instrument to tune:", ["Guitar", "Violin", "Piano"])

if st.button("Start Tuning"):
    CHUNK = 4096
    RATE = 44100
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    st.write(f"Listening for {instrument} notes... Play a note and follow the instructions.")
    
    tuning = TUNING[instrument]
    
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    try:
        while True:
            # Capture audio data
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            data = data / np.max(np.abs(data))  # Normalize
            
            # Detect frequency
            detected_frequency = detect_frequency(data, RATE)
            
            if detected_frequency > 20:  # Ignore noise
                closest_string, expected_frequency = find_closest_string(detected_frequency, tuning)
                feedback = provide_tuning_feedback(detected_frequency, closest_string, expected_frequency)
                
                st.write(f"Detected frequency: {detected_frequency:.2f} Hz")
                st.write(f"Closest note: {closest_string} ({expected_frequency} Hz)")
                st.write(feedback)
                st.write("-" * 50)
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()
        st.write("Tuning session ended.")
