import streamlit as st
import sounddevice as sd
import numpy as np
import io
from scipy.io.wavfile import write
from pydub import AudioSegment
from langchain_groq import ChatGroq


def generate_feedback(uploaded_text, recorded_text, instrument):
    """Generate feedback on mistakes and improvements using LLM."""
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama3-groq-70b-8192-tool-use-preview",
    )

    prompt = (
        f"Analyze the differences between the uploaded music sample and the user's recorded playing for the {instrument}. "
        f"Highlight mistakes or discrepancies and provide detailed feedback on how to improve playing skills."
        f"\nUploaded Sample:\n{uploaded_text}"
        f"\nUser's Recorded Sample:\n{recorded_text}"
    )
    response = llm.invoke(prompt)
    return response.content


# === Streamlit App ===
st.title("Play and Compare: Improve Your Skills")
st.subheader("Upload a melody, chord progression, or song you want to play and get feedback on your performance.")

# Step 1: Select Instrument
instrument = st.selectbox(
    "Select the instrument you are playing:",
    ["Piano", "Guitar", "Violin"]
)

# Step 2: Upload Melody or Song
uploaded_file = st.file_uploader("Upload a melody, chord progression, or song (WAV or MP3):", type=["wav", "mp3"])

if uploaded_file:
    # Display uploaded audio
    st.audio(uploaded_file, format=f"audio/{uploaded_file.type.split('.')[-1]}")
    uploaded_sample = AudioSegment.from_file(uploaded_file)
    uploaded_text = f"Duration: {len(uploaded_sample) / 1000:.2f} seconds"
    st.write(f"Uploaded Sample Details: {uploaded_text}")

    # Step 3: Record User's Attempt
    st.write("Now, play the uploaded song on your instrument.")

    # Initialize session state for recording
    if "recording_audio" not in st.session_state:
        st.session_state.recording_audio = None
        st.session_state.is_recording = False

    # Start Recording
    if not st.session_state.is_recording and st.button("Start Recording"):
        st.session_state.is_recording = True
        st.write("Recording started...")
        st.session_state.recording_audio = sd.rec(
            int(44100 * 60), samplerate=44100, channels=1, dtype="float32"
        )  # Record up to 60 seconds for safety

    # Stop Recording
    if st.session_state.is_recording and st.button("Stop Recording"):
        st.session_state.is_recording = False
        sd.stop()
        st.write("Recording stopped!")

    # Analyze and Compare
    if st.session_state.recording_audio is not None and not st.session_state.is_recording:
        st.subheader("Recorded Audio")

        # Convert the recording to WAV format
        audio_data = np.int16(st.session_state.recording_audio * 32767)  # Convert to int16 for WAV compatibility
        sample_rate = 44100  # Define sample rate

        # Save the audio to an in-memory buffer
        wav_buffer = io.BytesIO()
        write(wav_buffer, sample_rate, audio_data)
        wav_buffer.seek(0)

        # Play the recorded audio
        st.audio(wav_buffer, format="audio/wav")

        # Generate feedback if uploaded and recorded samples are available
        recorded_text = f"Recorded Sample Duration: {len(audio_data) / sample_rate:.2f} seconds"
        feedback = generate_feedback(uploaded_text, recorded_text, instrument)
        st.subheader("Feedback on Your Performance")
        st.text_area("Generated Feedback:", feedback, height=300)

