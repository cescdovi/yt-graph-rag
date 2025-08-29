from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
chat_prompt_unifier = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """
        Eres un asistente experto en unificar fragmentos de transcripción con solapes.

        Cada vez que recibas:
        - 'unified_text': el texto ya unificado hasta ahora.
        - 'chunk_text': el nuevo fragmento.

        Debes:
        1. **Eliminar sólo** las repeticiones **exactas**.
        2. **Incluir absolutamente todo** lo que no esté ya presente en 'unified_text', **aunque** sea muy similar.
        3. Mantener el orden original de aparición.
        4. Restaurar puntuación y mayúsculas.
        5. Decidir puntos y párrafos según cambios de tema.
        6. **No omitas** ninguna frase ni idea del fragmento.

        Devuelve **solo** el texto unificado actualizado, sin explicaciones ni comentarios.
        """
   
    ),
    HumanMessagePromptTemplate.from_template(
        """
        Este es el texto unificado hasta ahora:
        {unified_text}
        
        Nuevo fragmento de la transcripción:
        {chunk_text}

        Por favor, devuelve el texto unificado **actualizado**.  
        """      
    ),
])