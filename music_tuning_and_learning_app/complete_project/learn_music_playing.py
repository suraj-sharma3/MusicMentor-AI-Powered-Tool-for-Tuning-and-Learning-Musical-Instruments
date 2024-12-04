from langchain_groq import ChatGroq
import streamlit as st
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import tempfile


def generate_general_instructions(instrument):
    """
    Generate general instructions for playing the selected instrument using LLaMA.
    """
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama3-groq-70b-8192-tool-use-preview",
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
        groq_api_key="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",
        model_name="llama3-groq-70b-8192-tool-use-preview",
    )
    prompt = (
        f"Suggest suitable scales, chords, progressions, and melodies for playing {emotion} music on the {instrument}. "
        "Provide details on how to play these scales and chords."
    )
    response = llm.invoke(prompt)
    return response.content


def analyze_audio(data, rate, target_notes=None):
    """
    Analyze audio input to identify mistakes in pitch or timing.
    """
    fft_data = np.fft.rfft(data)
    frequencies = np.fft.rfftfreq(len(data), d=1.0 / rate)
    magnitude = np.abs(fft_data)

    # Identify the most prominent frequency
    peak_index = np.argmax(magnitude)
    peak_frequency = frequencies[peak_index]

    # Check pitch against target notes
    if target_notes:
        closest_note = min(target_notes, key=lambda note: abs(note - peak_frequency))
        if abs(closest_note - peak_frequency) > 5:  # Example threshold
            return f"Your pitch is off. Target frequency: {closest_note} Hz, Detected: {peak_frequency:.2f} Hz."
    return "Great job! Your pitch is on target."


def follow_up_question(previous_instructions, question):
    """
    Handle follow-up questions using LLaMA.
    """
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama3-groq-70b-8192-tool-use-preview",
    )
    prompt = f"""
    Previous Instructions:
    {previous_instructions}

    Follow-Up Question:
    {question}

    Provide a detailed answer to the question.
    """
    response = llm.invoke(prompt)
    return response.content


def compare_audio(uploaded_audio, recorded_audio, rate):
    """
    Compare the uploaded and recorded audio to identify differences.
    Handles stereo to mono conversion.
    """
    # Ensure both audios are mono
    if uploaded_audio.ndim > 1:
        uploaded_audio = np.mean(uploaded_audio, axis=1)  # Convert to mono
    if recorded_audio.ndim > 1:
        recorded_audio = np.mean(recorded_audio, axis=1)  # Convert to mono

    # Ensure both audios have the same length for comparison
    min_length = min(len(uploaded_audio), len(recorded_audio))
    uploaded_audio = uploaded_audio[:min_length]
    recorded_audio = recorded_audio[:min_length]

    # Compute the difference
    difference = np.abs(uploaded_audio - recorded_audio)
    pitch_mistakes = np.mean(difference)
    if pitch_mistakes > 0.1:  # Example threshold for significant difference
        return f"Significant pitch differences detected. Average deviation: {pitch_mistakes:.2f}"
    return "No significant differences detected. Great job!"


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

        # Step 4: Record Playback
        if st.button("Record Your Playback"):
            st.write("Recording... Play along with the uploaded sample.")
            duration = len(uploaded_audio) / rate  # Match the uploaded sample duration
            recorded_audio = sd.rec(int(duration * rate), samplerate=rate, channels=1, dtype="float32")
            sd.wait()

            # Save recorded audio for playback
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                wavfile.write(temp_file.name, rate, recorded_audio)
                st.audio(temp_file.name, format="audio/wav", start_time=0)
                st.success("Recorded your playback successfully!")

            # Step 5: Compare Audio
            st.write("Comparing your playback with the uploaded sample...")
            feedback = compare_audio(uploaded_audio, recorded_audio.flatten(), rate)
            st.markdown("### Feedback on Your Playback:")
            st.markdown(feedback)

            # Step 6: Follow-Up and Rectification
            rectification_steps = follow_up_question(feedback, "What steps can I take to improve?")
            st.markdown("### Steps to Rectify Your Mistakes:")
            st.markdown(rectification_steps)

            # Download options
            st.download_button(
                label="Download Feedback",
                data=feedback,
                file_name="playback_feedback.txt",
                mime="text/plain",
            )
            st.download_button(
                label="Download Rectification Steps",
                data=rectification_steps,
                file_name="rectification_steps.txt",
                mime="text/plain",
            )
