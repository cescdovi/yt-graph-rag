from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

CHAT_PROMPT_DETECT_LANGUAGE_SIMPLE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """
        Eres un clasificador de idioma. Responde excluisivamente sólo con el idioma principal del siguiente texto: valenciano, castellano u otro.
        """
    ),
    HumanMessagePromptTemplate.from_template(
        """
        Este es el texto introducido por el usuario:
        {text_to_dect}
        """
    )
])

CHAT_PROMPT_TRANSLATION_SIMPLE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """
        Eres un traductor experto que convierte valenciano al castellano manteniendo el estilo y la longitud del texto.
        """
    ),
    HumanMessagePromptTemplate.from_template(
        """
        Este es el texto introducido por el usuario:
        {text_to_translate}
        """
    )
])
