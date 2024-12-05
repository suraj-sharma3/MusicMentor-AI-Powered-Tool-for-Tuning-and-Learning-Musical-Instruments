from langchain_groq import ChatGroq

# LangChain setup for learning instructions
def generate_learning_instructions(instrument_or_question):
    """Generate instructions or answer questions about learning an instrument."""
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",
        model_name="llama3-groq-70b-8192-tool-use-preview",
    )

    prompt = f"Provide detailed, beginner-friendly instructions or answers for learning: {instrument_or_question}."
    response = llm.invoke(prompt)
    return response.content
