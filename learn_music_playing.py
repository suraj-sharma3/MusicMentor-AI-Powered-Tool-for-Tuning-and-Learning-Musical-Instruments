import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import sounddevice as sd
import streamlit as st
from langchain_groq import ChatGroq
import tempfile
import key_and_model


NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def frequency_to_note_name(freq):
    """
    Convert a frequency to the closest musical note.
    """
    if freq <= 0:
        return None
    A440 = 440.0  # Frequency of A4
    semitones = 12 * np.log2(freq / A440)
    note_index = int(round(semitones)) % 12
    return NOTE_NAMES[note_index]


def detect_dominant_frequency(audio_segment, rate):
    """
    Detect the dominant frequency in a segment of audio using FFT.
    """
    fft_data = np.fft.rfft(audio_segment)
    frequencies = np.fft.rfftfreq(len(audio_segment), d=1.0 / rate)
    magnitude = np.abs(fft_data)

    # Find the dominant frequency
    peak_index = np.argmax(magnitude)
    dominant_freq = frequencies[peak_index]
    return dominant_freq


def generate_general_instructions(instrument):
    """
    Generate general instructions for playing the selected instrument using LLaMA.
    """
    llm = ChatGroq(
        temperature=0,
        groq_api_key=key_and_model.groq_api_key,
        model_name=key_and_model.model,
    )
    prompt = f"Provide general instructions for a beginner to start learning to play the {instrument}."
    response = llm.invoke(prompt)
    return response.content


def generate_music_suggestions(emotion, instrument):
    """
    Generate music suggestions (scales, chords, progressions) based on emotion using LLaMA.
    """
    llm = ChatGroq(
        temperature=0,
        groq_api_key=key_and_model.groq_api_key,
        model_name=key_and_model.model,
    )
    prompt = (
        f"Suggest suitable scales, chords, progressions, and melodies for playing {emotion} music on the {instrument}. "
        "Provide details on how to play these scales and chords."
    )
    response = llm.invoke(prompt)
    return response.content


def generate_audio_comparison_suggestions(differences):
    """
    Use LLaMA to generate improvement suggestions based on audio comparison differences.
    """
    llm = ChatGroq(
        temperature=0,
        groq_api_key=key_and_model.groq_api_key,
        model_name=key_and_model.model,
    )
    prompt = f"Based on these audio differences, suggest detailed steps to improve: {differences}."
    response = llm.invoke(prompt)
    return response.content


def compare_audio(uploaded_audio, recorded_audio, rate):
    """
    Compare the uploaded and recorded audio to identify differences (pitch, timing, amplitude) with explanations.
    """
    if uploaded_audio.ndim > 1:
        uploaded_audio = np.mean(uploaded_audio, axis=1)
    if recorded_audio.ndim > 1:
        recorded_audio = np.mean(recorded_audio, axis=1)

    min_length = min(len(uploaded_audio), len(recorded_audio))
    uploaded_audio = uploaded_audio[:min_length]
    recorded_audio = recorded_audio[:min_length]

    segment_length = rate
    observations = []
    explanations = []

    for i in range(0, len(uploaded_audio), segment_length):
        uploaded_segment = uploaded_audio[i:i + segment_length]
        recorded_segment = recorded_audio[i:i + segment_length]

        # Detect dominant frequencies
        uploaded_freq = detect_dominant_frequency(uploaded_segment, rate)
        recorded_freq = detect_dominant_frequency(recorded_segment, rate)

        uploaded_note = frequency_to_note_name(uploaded_freq)
        recorded_note = frequency_to_note_name(recorded_freq)

        # Amplitude difference
        amplitude_diff = np.mean(np.abs(uploaded_segment - recorded_segment))

        # Timing difference: Compare when the peak amplitude occurs
        uploaded_peak_time = np.argmax(uploaded_segment) / rate
        recorded_peak_time = np.argmax(recorded_segment) / rate
        timing_diff = abs(uploaded_peak_time - recorded_peak_time)

        observations.append({
            "Time (s)": f"{i // rate}-{(i + segment_length) // rate}",
            "Uploaded Note": uploaded_note or "N/A",
            "Recorded Note": recorded_note or "N/A",
            "Amplitude Difference": f"{amplitude_diff:.2f}",
            "Timing Difference (s)": f"{timing_diff:.2f}",
            "Pitch Match": "Yes" if uploaded_note == recorded_note else "No"
        })

        # Generate explanations
        if uploaded_note == recorded_note:
            explanations.append(
                f"In the interval {i // rate}-{(i + segment_length) // rate} seconds, the notes match."
            )
        else:
            explanations.append(
                f"In the interval {i // rate}-{(i + segment_length) // rate} seconds, the uploaded note was "
                f"{uploaded_note}, but the recorded note was {recorded_note}. "
                f"Amplitude difference: {amplitude_diff:.2f}, Timing difference: {timing_diff:.2f}s. "
                "Consider revisiting this section."
            )

    return observations, explanations


def learn_music_playing_ui():
    """
    Streamlit UI for the Learn Music by Playing feature.
    """
    st.write("Welcome to the Learn Music by Playing page!")

    # Step 1: Select Instrument
    instrument = st.selectbox("Select your instrument:", ["Guitar", "Violin", "Piano"])
    if st.button("Get General Instructions"):
        general_instructions = generate_general_instructions(instrument)
        st.markdown("### General Instructions:")
        st.markdown(general_instructions)
        st.download_button(
            label="Download General Instructions",
            data=general_instructions,
            file_name=f"general_instructions_{instrument}.txt",
            mime="text/plain",
        )

    # Step 2: Select Emotion
    emotion = st.radio("What type of music do you want to play?", ["Happy", "Sad", "Relaxing", "Energetic"])
    if st.button("Get Music Suggestions"):
        music_suggestions = generate_music_suggestions(emotion, instrument)
        st.markdown("### Music Suggestions:")
        st.markdown(music_suggestions)
        st.download_button(
            label="Download Music Suggestions",
            data=music_suggestions,
            file_name=f"music_suggestions_{emotion}_{instrument}.txt",
            mime="text/plain",
        )

    # Step 3: Upload Music Sample
    uploaded_file = st.file_uploader("Upload a music sample (WAV format):", type=["wav"])
    if uploaded_file:
        rate, uploaded_audio = wavfile.read(uploaded_file)
        st.audio(uploaded_file, format="audio/wav", start_time=0)
        st.success("Uploaded music sample successfully!")

        # Step 4: Start/Stop Recording Playback
        if "recording" not in st.session_state:
            st.session_state["recording"] = False

        if st.session_state["recording"]:
            if st.button("Stop Recording"):
                st.session_state["recording"] = False
                sd.stop()
                recorded_audio = st.session_state.get("recorded_audio")

                # Save recorded audio for playback
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    wavfile.write(temp_file.name, rate, recorded_audio)
                    st.audio(temp_file.name, format="audio/wav", start_time=0)
                    st.success("Recording stopped. Analyzing your playback...")

                # Compare Audio
                differences, explanations = compare_audio(uploaded_audio, recorded_audio.flatten(), rate)

                # Display Differences
                st.markdown("### Audio Differences:")
                differences_df = pd.DataFrame(differences)
                st.table(differences_df)

                # Display Explanations
                st.markdown("### Explanations of Differences:")
                for explanation in explanations:
                    st.write(f"- {explanation}")

                # Generate and Display Suggestions
                suggestions = generate_audio_comparison_suggestions(differences)
                st.markdown("### Suggestions for Improvement:")
                st.markdown(suggestions)
                st.download_button(
                    label="Download Suggestions",
                    data=suggestions,
                    file_name="audio_comparison_suggestions.txt",
                    mime="text/plain",
                )
        else:
            if st.button("Start Recording"):
                st.session_state["recording"] = True
                st.write("Recording... Play along with the uploaded sample.")
                duration = 30
                st.session_state["recorded_audio"] = sd.rec(int(duration * rate), samplerate=rate, channels=1, dtype="float32")
