from langchain_groq import ChatGroq
import streamlit as st

# LangChain setup for tuning instructions
def generate_tuning_instructions(instrument_or_question):
    """Generate instructions or answer questions about tuning."""
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama3-groq-70b-8192-tool-use-preview",
    )

    prompt = f"Provide detailed, beginner-friendly instructions or answers for tuning: {instrument_or_question}."
    response = llm.invoke(prompt)
    return response.content

def assist_tuning_guitar_or_violin(instrument):
    """Assist users in tuning guitar or violin step by step."""
    if instrument == "Guitar":
        strings = ["E (low)", "A", "D", "G", "B", "E (high)"]
    elif instrument == "Violin":
        strings = ["G", "D", "A", "E"]
    else:
        st.error(f"Tuning assistance is not available for {instrument}.")
        return

    for string in strings:
        st.write(f"Play the {string} string on your {instrument}.")
        st.write("Ensure it matches the correct pitch.")
        if st.button(f"Mark {string} as tuned", key=f"{string}_tuned"):
            st.success(f"{string} string is now tuned!")
