from langchain.prompts import ChatPromptTemplate

CHAT_PROMPT_STRUCTURED_OUTPUTS_SIMPLE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un experto en análisis semántico y extracción de información estructurada a partir de texto natural.\n"
            "Tienes que identificar entidades y relaciones en un texto proporcionado, y devolver la información en formato JSON válido, siguiendo exactamente el esquema pydantic que se te proporciona a continuación.\n"
            "Asegúrate de cumplir las siguientes instrucciones:\n"
            "Tu respuesta debe estar *únicamente* en formato JSON y envuelta dentro de triple backticks (```).\n"
            "Sigue estrictamente este esquema:"
            "{format_instructions}\n\n"
            "Si no se encuentran entidades o relaciones en el texto, devuelve objetos vacíos según el esquema."
        ),
        ("human", "{text_to_extract_entities}"),
    ]
)