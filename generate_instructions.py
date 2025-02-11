from langchain_groq import ChatGroq
import streamlit as st
import key_and_model



def generate_tuning_instructions(instrument, skill_level):
    """
    Generate tuning instructions for a specific instrument and skill level using LLaMA.
    """
    llm = ChatGroq(
        temperature=0,
        groq_api_key=key_and_model.groq_api_key,
        model_name=key_and_model.model,
    )

    prompt = (
        f"Provide detailed, beginner-friendly instructions for tuning a {instrument}."
        if skill_level == "Beginner"
        else f"Provide detailed, {skill_level.lower()}-level instructions for tuning a {instrument}."
    )

    response = llm.invoke(prompt)
    return response.content


def generate_learning_instructions(instrument, skill_level):
    """
    Generate learning instructions for a specific instrument and skill level using LLaMA.
    """
    llm = ChatGroq(
        temperature=0,
        groq_api_key=key_and_model.groq_api_key,
        model_name=key_and_model.model,
    )

    prompt = (
        f"Provide detailed, beginner-friendly learning instructions for playing a {instrument}."
        if skill_level == "Beginner"
        else f"Provide detailed, {skill_level.lower()}-level learning instructions for playing a {instrument}."
    )

    response = llm.invoke(prompt)
    return response.content


def follow_up_question(previous_instructions, question):
    """
    Handle follow-up questions using the LLaMA model.
    """
    llm = ChatGroq(
        temperature=0,
        groq_api_key=key_and_model.groq_api_key,
        model_name=key_and_model.model,
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


def generate_tuning_and_learning_instructions():
    """
    Streamlit interface for generating tuning and learning instructions based on skill level.
    """
    st.write("Use this page to generate personalized tuning and learning instructions.")

    # Inputs for instrument and skill level
    instruction_type = st.radio("Choose instruction type:", ["Tuning", "Learning"])
    instrument = st.selectbox("Select an instrument:", ["Guitar", "Violin", "Piano"])
    skill_level = st.radio("Select your skill level:", ["Beginner", "Intermediate", "Expert"])

    # Generate initial instructions
    if st.button("Generate Instructions"):
        if instruction_type == "Tuning":
            st.session_state["instructions"] = generate_tuning_instructions(instrument, skill_level)
        else:
            st.session_state["instructions"] = generate_learning_instructions(instrument, skill_level)

        st.session_state["instruction_type"] = instruction_type
        st.session_state["instrument"] = instrument

    # Display generated instructions if available
    if "instructions" in st.session_state:
        instructions = st.session_state["instructions"]
        instruction_type = st.session_state["instruction_type"]
        instrument = st.session_state["instrument"]

        st.markdown(f"### Generated {instruction_type} Instructions:")
        st.markdown(instructions)

        # Add options to download and copy the instructions
        st.download_button(
            label="Download Instructions",
            data=instructions,
            file_name=f"{instruction_type}_instructions_{instrument}.txt",
            mime="text/plain",
        )

        st.code(instructions, language="markdown")

        # Follow-up questions
        st.markdown("### Ask Follow-Up Questions:")
        follow_up = st.text_area("Enter your follow-up question here:", key="follow_up_question")

        if st.button("Submit Follow-Up Question"):
            if follow_up.strip():
                st.session_state["follow_up_response"] = follow_up_question(instructions, follow_up)

    # Display follow-up response if available
    if "follow_up_response" in st.session_state:
        follow_up_response = st.session_state["follow_up_response"]
        st.markdown("### Follow-Up Response:")
        st.markdown(follow_up_response)

        # Allow users to download follow-up responses
        st.download_button(
            label="Download Follow-Up Response",
            data=follow_up_response,
            file_name=f"{instruction_type}_follow_up_{instrument}.txt",
            mime="text/plain",
        )

        # Display follow-up response in a copyable format
        st.code(follow_up_response, language="markdown")
