from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

chat_prompt_transcription = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Eres un LLM experto en transcribir fragmentos de audio con solapamiento."
    ),
    HumanMessagePromptTemplate.from_template(
        "Archivo: {chunk_filename}\n"
        "Transcribe manteniendo coherencia entre los 1 min de solapamiento."
    ),
])