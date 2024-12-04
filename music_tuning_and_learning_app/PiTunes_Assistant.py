import streamlit as st
from tuning import generate_tuning_instructions
from learning import generate_learning_instructions

# Set up Streamlit app
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

# Option to download and copy initial instructions
if instructions:
    st.download_button(
        label="Download Instructions",
        data=instructions,
        file_name=f"{instrument.lower()}_{action.lower()}_instructions.txt",
        mime="text/plain",
    )
    if st.button("Copy Instructions to Clipboard"):
        st.experimental_set_query_params(instructions=instructions)
        st.success("Instructions copied to clipboard!")

# Option to download and copy follow-up response
if follow_up_response:
    st.download_button(
        label="Download Follow-up Answer",
        data=follow_up_response,
        file_name=f"{instrument.lower()}_{action.lower()}_follow_up.txt",
        mime="text/plain",
    )
    if st.button("Copy Follow-up Answer to Clipboard"):
        st.experimental_set_query_params(follow_up_response=follow_up_response)
        st.success("Follow-up answer copied to clipboard!")
