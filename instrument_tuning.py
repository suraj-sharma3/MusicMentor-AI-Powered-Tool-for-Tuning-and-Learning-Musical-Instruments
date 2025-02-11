import streamlit as st
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import time

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
        # Continue full dictionary...
    },
}

def tune_instrument_ui():
    """
    Streamlit UI for tuning an instrument.
    """
    st.write("Use this page to tune your instrument.")

    # Select instrument
    instrument = st.selectbox("Select an instrument:", ["Guitar", "Violin", "Piano"])
    start_button = st.button("Start Tuning")
    stop_button = st.button("Stop Tuning")

    if start_button:
        st.write(f"Tuning your {instrument}...")
        tuning = TUNING[instrument]

        CHUNK = 4096
        RATE = 44100
        FORMAT = pyaudio.paInt16
        CHANNELS = 1

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

        recording_active = True

        # Streamlit placeholders for live updates
        waveform_placeholder = st.empty()
        details_placeholder = st.empty()

        while recording_active:
            # Read audio data
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            data = data / np.max(np.abs(data))  # Normalize

            # Detect frequency (logic encapsulated in modular tuning code)
            detected_frequency = detect_frequency(data, RATE)

            # Generate waveform visualization
            fig, ax = plt.subplots()
            ax.plot(data, color="blue")
            ax.set_title("Audio Waveform")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude")
            waveform_placeholder.pyplot(fig)

            # Provide feedback
            if detected_frequency > 20:  # Ignore noise
                closest_string, expected_frequency = find_closest_string(detected_frequency, tuning)
                feedback = provide_tuning_feedback(detected_frequency, closest_string, expected_frequency)

                details_placeholder.markdown(
                    f"""
                    **Detected frequency:** {detected_frequency:.2f} Hz  
                    **Closest note:** {closest_string} ({expected_frequency} Hz)  
                    **Feedback:** {feedback}
                    """
                )

            if stop_button:
                recording_active = False
                st.write("Tuning stopped.")

        stream.stop_stream()
        stream.close()
        p.terminate()

# Functions reused for modularity
def detect_frequency(data, rate):
    fft_data = np.fft.rfft(data)
    frequencies = np.fft.rfftfreq(len(data), d=1.0 / rate)
    magnitude = np.abs(fft_data)
    peak_index = np.argmax(magnitude)
    return frequencies[peak_index]

def find_closest_string(frequency, tuning):
    return min(tuning, key=lambda key: abs(tuning[key] - frequency)), tuning[min(tuning, key=lambda key: abs(tuning[key] - frequency))]

def provide_tuning_feedback(detected_frequency, closest_string, expected_frequency):
    difference = detected_frequency - expected_frequency
    if abs(difference) < 1:
        return f"{closest_string} is in tune! 🎵"
    elif difference > 0:
        return f"{closest_string} is sharp (too high). Loosen the string slightly."
    else:
        return f"{closest_string} is flat (too low). Tighten the string slightly."
