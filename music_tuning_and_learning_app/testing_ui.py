import streamlit as st
import numpy as np
import sounddevice as sd
import io
from scipy.io.wavfile import write
from pydub import AudioSegment
from langchain_groq import ChatGroq
from tuning import generate_tuning_instructions, assist_tuning_guitar_or_violin
from learning import generate_learning_instructions


# === Helper Functions ===
def generate_feedback(uploaded_text, recorded_text, instrument):
    """Generate feedback on mistakes and improvements."""
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


def generate_piano_insights():
    """Generate LLM-based insights about electronic and acoustic pianos."""
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama3-groq-70b-8192-tool-use-preview",
    )
    prompt = (
        "Explain why an electronic piano doesn't need tuning and why an acoustic piano should be tuned by an expert."
    )
    response = llm.invoke(prompt)
    return response.content


# === Main Page for Tuning and Learning Assistance ===
def main_page():
    """Main page for tuning and learning assistance."""
    st.title("HarmonyPi: Instrument Assistant")
    st.subheader("Get guidance for tuning or learning your instrument.")

    # Dropdown to select an instrument
    instrument = st.selectbox(
        "Select the instrument you want help with:",
        ["Piano", "Guitar", "Violin"]
    )

    # Radio buttons to choose between tuning or learning
    action = st.radio(
        f"What would you like to do with the {instrument}?",
        ["Tuning", "Learning"]
    )

    # Placeholder for instructions
    instructions = ""

    # Generate initial instructions
    if st.button("Get Instructions"):
        st.write(f"Generating {action.lower()} instructions for the {instrument}...")
        try:
            if action == "Tuning":
                instructions = generate_tuning_instructions(instrument)
            elif action == "Learning":
                instructions = generate_learning_instructions(instrument)
            st.text_area(
                "Generated Instructions:",
                instructions,
                height=300,
                key="instructions_area",
            )
        except Exception as e:
            st.error(f"An error occurred while generating instructions: {e}")

    # Allow follow-up questions
    question = st.text_input("Have more questions? Ask here:")
    follow_up_response = ""

    if st.button("Submit Question"):
        if question.strip():
            st.write(f"Answering your question: {question}")
            try:
                if action == "Tuning":
                    follow_up_response = generate_tuning_instructions(f"{instrument}: {question}")
                elif action == "Learning":
                    follow_up_response = generate_learning_instructions(f"{instrument}: {question}")
                st.text_area(
                    "Follow-up Answer:",
                    follow_up_response,
                    height=200,
                    key="follow_up_area",
                )
            except Exception as e:
                st.error(f"An error occurred while answering your question: {e}")
        else:
            st.warning("Please enter a question before submitting.")


# === Audio Recording Page ===
def audio_recording_page():
    """Page for recording audio and assisting with tuning or insights."""
    st.title("Record and Analyze Your Instrument")
    st.subheader("Record yourself playing your selected instrument and get insights or tuning assistance.")

    instrument = st.selectbox(
        "Select the instrument you are playing:",
        ["Piano", "Guitar", "Violin"]
    )

    # Initialize session state for recording
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
        st.session_state.audio_data = None

    # Start Recording
    if not st.session_state.is_recording and st.button("Start Recording"):
        st.session_state.is_recording = True
        st.write("Recording started...")
        st.session_state.audio_data = sd.rec(
            int(44100 * 60), samplerate=44100, channels=1, dtype="float32"
        )  # Recording up to 60 seconds for safety

    # Stop Recording
    if st.session_state.is_recording and st.button("Stop Recording"):
        st.session_state.is_recording = False
        sd.stop()
        st.write("Recording stopped!")

    # Display Waveform and Play Audio
    if st.session_state.audio_data is not None and not st.session_state.is_recording:
        st.subheader("Recorded Audio Details")
        st.audio(st.session_state.audio_data.tobytes(), format="audio/wav")

        # Provide insights or tuning assistance
        if instrument == "Piano":
            st.subheader("Insights for Piano")
            piano_insights = generate_piano_insights()
            st.write(piano_insights)
        else:
            st.subheader("Tuning Assistance")
            assist_tuning_guitar_or_violin(instrument)


# === Upload and Compare Melody Page ===
def melody_comparison_page():
    """Page for uploading a melody and comparing it with the user's attempt."""
    st.title("Melody Comparison and Feedback")
    st.subheader("Upload a melody or chord progression and get feedback on your performance.")

    instrument = st.selectbox(
        "Select the instrument you are playing:",
        ["Piano", "Guitar", "Violin"]
    )

    uploaded_file = st.file_uploader("Upload a melody or chord progression (WAV or MP3):", type=["wav", "mp3"])

    # Display uploaded file details
    if uploaded_file:
        st.audio(uploaded_file, format=f"audio/{uploaded_file.type.split('.')[-1]}")
        uploaded_sample = AudioSegment.from_file(uploaded_file)
        uploaded_text = f"Duration: {len(uploaded_sample) / 1000:.2f} seconds"
        st.write(f"Uploaded Sample Details: {uploaded_text}")

    # Initialize session state for recording
    if "comparison_audio" not in st.session_state:
        st.session_state.comparison_audio = None
        st.session_state.is_comparison_recording = False

    # Start recording
    if uploaded_file and not st.session_state.is_comparison_recording and st.button("Start Recording"):
        st.session_state.is_comparison_recording = True
        st.write("Recording started...")
        st.session_state.comparison_audio = sd.rec(
            int(44100 * 60), samplerate=44100, channels=1, dtype="float32"
        )  # Recording up to 60 seconds for safety

    # Stop recording
    if st.session_state.is_comparison_recording and st.button("Stop Recording"):
        st.session_state.is_comparison_recording = False
        sd.stop()
        st.write("Recording stopped!")

    # Analyze and Compare
    if st.session_state.comparison_audio is not None and not st.session_state.is_comparison_recording:
        st.subheader("Recorded Audio Details")
        st.audio(st.session_state.comparison_audio.tobytes(), format="audio/wav")

        # Generate feedback if uploaded and recorded samples are available
        if uploaded_file:
            recorded_text = f"Recorded Sample Duration: {len(st.session_state.comparison_audio) / 44100:.2f} seconds"
            feedback = generate_feedback(uploaded_text, recorded_text, instrument)
            st.subheader("Feedback on Your Performance")
            st.text_area("Generated Feedback:", feedback, height=300)


# === Streamlit Navigation ===
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Record & Analyze", "Upload & Compare"])

if page == "Home":
    main_page()
elif page == "Record & Analyze":
    audio_recording_page()
elif page == "Upload & Compare":
    melody_comparison_page()
