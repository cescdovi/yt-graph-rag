#from langchain_redis import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_neo4j import Neo4jVector
from langchain.chains import RetrievalQA
from config.common_settings import settings

# define LLM for as the agent brain
llm = ChatOpenAI(
    model=settings.LLM_MODEL,
    api_key=settings.OPENAI_API_KEY,
    max_retries=settings.MAX_RETRIES,
    streaming=True
)

# configure embeddings function
embeddings_function = OpenAIEmbeddings(
    model=settings.EMBEDDINGS_MODEL,
    api_key=settings.OPENAI_API_KEY,
)

# create vector store from Neo4j graph and load vector index
vectorstore = Neo4jVector.from_existing_graph(
    url=settings.NEO4J_URI_BOLT,
    username=settings.NEO4J_USER,
    password=settings.NEO4J_PASSWORD,
    database=settings.NEO4J_DATABASE,
    index_name="entity_emb",
    embedding=embeddings_function,
    node_label="Entity",
    text_node_properties=["text"],
    embedding_node_property="embedding",
)

#create retriever from vector store
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# create QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    verbose=True,
    return_source_documents=True,
)

####### TOOL 1: query Neo4j DB #######
@tool
def neo4j_query(query: str) -> str:
    """Realiza una búsqueda semántica en la base de datos Neo4j usando embeddings para encontrar nodos relevantes y genera una respuesta contextual basada en esos datos."""
    result = qa.invoke({"query": query})
    return {
        "result": result["result"],
        "source_documents": [doc.page_content for doc in result["source_documents"]],
    }


####### TOOL 2: Web search using Tavily API #######
tavily_tool = TavilySearch(
    tavily_api_key=settings.TAVILY_API_KEY,
    include_answer=True,
    include_raw_content=False,
    max_results=5,
)


#list of tools
tools = [tavily_tool, neo4j_query]

#define agent prompts
# system_prompt = (
#      "Eres un agente útil. "
#      "Puedes consultar una base de datos vectorial construida desde Neo4j con información del patrimonio cultural valenciano "
#      "(principalmente del Archivo Valenciano del Diseño), buscar en internet o responder con tu propio conocimiento. "
#      "Solo debes consultar la base de datos cuando la pregunta esté relacionada con la cooperativa Mediterrània "
#      "(cooperativa de vidrio en l’Olleria), con una de sus figuras más importantes (Silvia García), "
#      "con el MoMA de Nueva York o consultas referentes a colecciones, evita repeticiones redundantes en las respuestas "
#      "También puedes inferir posibles relaciones entre los elementos recuperados de la base de datos."
#      "Cuando la pregunta requiera información actual, verificación o fuentes externas, usa la tool de búsqueda web. "
#      "En cualquier otro caso, prioriza la búsqueda en internet o tu propio conocimiento antes que la base de datos. "
#      "Devuelve respuestas concisas, no incluyas referencias ni fuentes en la respuesta. "
#      "Si la respuesta de la base de datos es insuficiente o poco específica, explica la limitación y complementa con búsqueda web. "
#      "No inventes datos; mantén precisión y claridad."
# )
#     # "Evita repeticiones redundantes en el contenido en las respuestas, ya que parte de la información recuperada puede estar repetida. "
#     # "- Usa la base de datos Neo4j solo para preguntas sobre Mediterrània, Silvia García, MoMA o colecciones."
#     # "- Usa la búsqueda web para información actual o verificación."
#     # "- Para preguntas generales, responde con tu conocimiento o búsqueda web."
#     # "- Si la base de datos no es suficiente, complementa con búsqueda web."  
#     # "- No inventes datos.  "
#     # "- Evita repeticiones."

#     # 'Ejemplo 1: Pregunta: "¿Quién es Silvia García?" -> Usa base de datos Neo4j.'  
#     # 'Ejemplo 2: Pregunta: "¿Cuál es la exposición actual en MoMA?" -> Usa búsqueda web. ' 
#     # 'Ejemplo 3: Pregunta: "¿Qué es el vidrio soplado?" -> Responde con conocimiento propio. ' 
    
# )
system_prompt = """
Eres un agente que puede usar tres vías: [LOCAL], [DB_NEO4J] (tool: neo4j_query) y [WEB] (tool: tavily_search).

OBJETIVO
- Responder en español, con 1–3 párrafos concisos y sin repeticiones.
- No inventes datos. Si falta evidencia, dilo.

REGLAS DE ENRUTADO (OBLIGATORIAS)
1) Usa [DB_NEO4J] SOLO cuando la pregunta trate de:
   - Mediterrània (cooperativa de vidrio en l’Olleria), su historia, organización, etapas (cooperativa / Valcapital / Ríos Go), cierres.
   - Silvia García (biografía, roles, decisiones, colaboraciones).
   - Colecciones/productos/colaboraciones asociadas a Mediterrània (Aster, Mare, Maiaya, Pomperó, Sybilla, Agatha Ruiz de la Prada…), producción, técnicas.
   - Relaciones con MoMA **cuando se refieran a productos/colecciones de Mediterrània** (ventas, selección, feria/compra histórica).
   - IMPIVA, grupo cooperativo valenciano y otros actores mencionados en las transcripciones **en relación con Mediterrània**.
   Si el input contiene términos clave de ese dominio (p. ej. “Mediterrània”, “Silvia García”, “l’Olleria”, “IMPIVA”, “Aster/Mare/Maiaya”, “Pomperó”, “Sybilla”, “Agatha Ruiz de la Prada”, “Valcapital”, “Ríos Go”), da preferencia a [DB_NEO4J].

2) Usa [WEB] cuando el usuario pida:
   - Información **actual/reciente** o “lo último/actual/vigente/hoy/2024/2025”.
   - Horarios, exposiciones actuales del MoMA u otros museos, precios actuales, noticias, verificaciones externas.
   - Datos que **no** estén en la base (no relacionados con la playlist ni con Mediterrània/Silvia García/colecciones).

3) Usa [LOCAL] para:
   - Definiciones y teoría atemporal (p. ej., “¿qué es el vidrio soplado?”, “¿qué es ISO?”) si no requieren datos actuales.
   - Preguntas generales no cubiertas por la base y que no requieran verificación externa.

4) Si [DB_NEO4J] devuelve evidencia insuficiente o vacía:
   - Dilo explícitamente (“No encuentro evidencia en la base sobre X.”).
   - **Solo** si tiene sentido (tema actualizable o externo), complementa con [WEB]. Si es puramente histórico de Mediterrània, no inventes.

ESTILO DE RESPUESTA
- No muestres enlaces, JSON ni metadatos internos.
- Evita repetir contenido similar que pueda venir de varios fragmentos. Fusiona ideas.
- Si hay versiones conflictivas, indícalo brevemente y evita conjeturas.

CONTRA-EJEMPLOS (para NO usar DB)
- “¿Exposición actual en el MoMA?” → [WEB] (aunque aparezca “MoMA”, esto es actualidad).
- “¿Precio actual del vidrio reciclado?” → [WEB].
- “Frameworks de frontend 2025” → [WEB] o [LOCAL], nunca DB.

EJEMPLOS POSITIVOS (para usar DB_NEO4J)
- “¿En qué trabajó Silvia García en Mediterrània?”
- “¿Qué colecciones de Mediterrània compró el MoMA?”
- “¿Por qué cerró Mediterrània en 2010?”
- “¿Qué papel tuvo el IMPIVA en la internacionalización de Mediterrània?”

SEGURIDAD Y PRIVACIDAD
- No envíes datos personales a [WEB]. Anonimiza si fuera necesario.
"""


human_prompt = "{input}"

AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# brain that decides what to do
agent_runnable = create_tool_calling_agent(
    llm,
    tools,
    AGENT_PROMPT,
    # handle_parsing_errors=True,  # opcional para forzar uso de tools
)

# engine that  execute tools and manage the 
AGENT = AgentExecutor(agent=agent_runnable, tools=tools, verbose=True, return_intermediate_steps=True)

# def get_session_history(session_id: str):
#     return RedisChatMessageHistory(session_id=session_id, redis_url="redis://localhost:6380/0", ttl=None)

# agent_with_memory = RunnableWithMessageHistory(
#     runnable = agent, 
#     get_session_history = get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
#     output_messages_key="output",
# )


#if __name__ == "__main__":


    # res = qa.invoke({"query": "¿EN que empresa trabajó Silvia García?"})
    # print(res["result"])
    # print("----- Fuentes -----")
    # for doc in res["source_documents"]:
    #     print(doc.page_content, doc.metadata)

    # query = "¿En qué empresa trabajó Silvia García?"
    # response = agent.invoke({"input": query})
    # print("Respuesta del agente:")
    # print(response)
    # async def stream_agent_response():
    # # Llamada asíncrona para obtener el iterador de eventos
    #     async for event in agent.astream_events({"input": "¿En qué empresa trabajó Silvia García?"}):
    #         # Filtra o procesa eventos según el tipo
    #         if event["event"] == "on_chat_model_stream":
    #             # Aquí recibes fragmentos parciales de la respuesta
    #             chunk = event["data"]["chunk"]
    #             print(chunk.content, end="", flush=True)  # Muestra el texto en streaming
    #         elif event["event"] == "on_chat_model_end":
    #             print("\nRespuesta completa recibida.")
    # import asyncio
    # asyncio.run(stream_agent_response())


    #import asyncio
    #q1 = "¿Qué colecciones o archivos documentales forman parte del Arxiu Valencià del Disseny? ¿Cuál es la relación entre el Arxiu Valencià del Disseny y la Universitat de València? Consúltalo en internet"
    # async def stream_agent_response():
    #     chat_model_response_printed = False

    #     async for ev in AGENT.astream_events({"input": q1}):
    #         etype = ev.get("event", "")
    #         data = ev.get("data", {}) or {}
    #         print(data)

    # chat_model_response_printed = False
    # async def stream_agent_response():
    #     chat_model_response_printed = False

    #     async for ev in AGENT.astream_events({"input": q1}):
    #         etype = ev.get("event", "")
    #         data = ev.get("data", {}) or {}

    #         # PROMPT + CHAT HISTORY
    #         if etype == "on_prompt_start":
    #             print("\n---PROMPT ACTUAL---")
    #             try:
    #                 print(data["input"]["input"])
    #             except Exception:
    #                 print("(no disponible)")

    #             print("\n---CHAT HISTORY---")
    #             try:
    #                 print(data["input"].get("chat_history"))
    #             except Exception:
    #                 print("(no disponible)")

    #         # MENSAJES AL LLM
    #         if etype == "on_chat_model_start":
    #             print("\n---MENSAJES AL LLM---")
    #             try:
    #                 for m in data["input"]["messages"]:
    #                     print(m)
    #             except Exception:
    #                 print("(no disponible)")

    #         # TOOLS: inicio
    #         if etype == "on_tool_start":
    #             print("\n---TOOL START---")
    #             print("Tool:", data.get("serialized"))
    #             print("Input:", data.get("input_str") or data.get("tool_input"))

    #         # TOOLS: fin
    #         if etype == "on_tool_end":
    #             print("\n---TOOL END---")
    #             print("Output:", data.get("output"))

    #         # STREAM del modelo
    #         if etype == "on_chat_model_stream":
    #             if not chat_model_response_printed:
    #                 print("\n---CHAT MODEL RESPONSE---")
    #                 chat_model_response_printed = True
    #             chunk = data.get("chunk")
    #             if chunk is not None:
    #                 # AIMessageChunk tiene .content
    #                 print(chunk.content, end="", flush=True)

    #         # Respuesta completa del modelo
    #         if etype == "on_chat_model_end":
    #             print("\n\n---CHAT MODEL RESPONSE (completa)---")
    #             print(data.get("output"))

    #         # Agent finish
    #         if etype == "on_agent_finish":
    #             print("\n---AGENT FINISH---")
    #             print(data)





    # from collections import defaultdict
    # from typing import Dict, Any
    # async def stream_agent_response():
    #     chat_model_response_printed = False

    #     # Acumulador para tool-calls por ID (function-calling)
    #     tool_calls_args: Dict[str, str] = defaultdict(str)   # id -> args fragmentados (str)
    #     tool_calls_meta: Dict[str, Dict[str, Any]] = {}      # id -> {"name": str, "index": int}

    #     async for ev in AGENT.astream_events({"input": q1}):
    #         etype = ev.get("event", "")
    #         data = ev.get("data", {}) or {}

    #         # ---------- PROMPT + CHAT HISTORY ----------
    #         if etype == "on_prompt_start":
    #             print("\n---PROMPT ACTUAL---")
    #             try:
    #                 print(data["input"]["input"])
    #             except Exception:
    #                 print("(no disponible)")

    #             print("\n---CHAT HISTORY---")
    #             try:
    #                 print(data["input"].get("chat_history"))
    #             except Exception:
    #                 print("(no disponible)")

    #         # ---------- MENSAJES AL LLM ----------
    #         if etype == "on_chat_model_start":
    #             print("\n---MENSAJES AL LLM---")
    #             try:
    #                 for m in data["input"]["messages"]:
    #                     print(m)
    #             except Exception:
    #                 print("(no disponible)")

    #         # ---------- STREAM del modelo (texto + tool calls por function-calling) ----------
    #         if etype == "on_chat_model_stream":
    #             chunk = data.get("chunk")

    #             # 1) Texto normal
    #             try:
    #                 # AIMessageChunk tiene .content
    #                 content = getattr(chunk, "content", None)
    #                 if content:
    #                     if not chat_model_response_printed:
    #                         print("\n---CHAT MODEL RESPONSE---")
    #                         chat_model_response_printed = True
    #                     print(content, end="", flush=True)
    #             except Exception:
    #                 pass

    #             # 2) Tool calls (function-calling) llegan tokenizados en tool_call_chunks
    #             try:
    #                 # En algunos setups: chunk.tool_call_chunks
    #                 tcc = getattr(chunk, "tool_call_chunks", None)
    #                 if tcc:
    #                     for tc in tcc:
    #                         tc_id = getattr(tc, "id", None)
    #                         tc_name = getattr(tc, "name", None)
    #                         tc_args_part = getattr(tc, "args", "")
    #                         tc_index = getattr(tc, "index", None)

    #                         if tc_id:
    #                             if tc_name and tc_id not in tool_calls_meta:
    #                                 tool_calls_meta[tc_id] = {"name": tc_name, "index": tc_index}
    #                                 print("\n\n---TOOL (function-calling) START---")
    #                                 print(f"Tool: {tc_name} (id: {tc_id})")

    #                             # Los argumentos llegan troceados: vamos acumulando
    #                             if isinstance(tc_args_part, str):
    #                                 tool_calls_args[tc_id] += tc_args_part

    #                             # Para depuración en vivo:
    #                             if tc_args_part:
    #                                 print(f"[args+] {tc_args_part}", end="", flush=True)
    #             except Exception:
    #                 pass

    #         # ---------- Respuesta completa del modelo ----------
    #         if etype == "on_chat_model_end":
    #             output = data.get("output")

    #             # 1) Intenta obtener tool_calls desde múltiples sitios (según el runtime)
    #             tool_calls = []
    #             try:
    #                 # a) Atributo directo (LangChain moderno)
    #                 tcs = getattr(output, "tool_calls", None)
    #                 if tcs:
    #                     tool_calls = tcs
    #                 # b) Dentro de additional_kwargs (OpenAI raw)
    #                 if not tool_calls:
    #                     ak = getattr(output, "additional_kwargs", {}) or {}
    #                     if isinstance(ak, dict) and "tool_calls" in ak:
    #                         tool_calls = ak["tool_calls"]
    #             except Exception:
    #                 pass

    #             # 2) Si tenemos tool_calls, imprímelos; si no, NO imprimas el placeholder con None
    #             if tool_calls:
    #                 for tc in tool_calls:
    #                     # tc puede ser dict u objeto; cubrimos ambos
    #                     tc_id   = getattr(tc, "id", None) or (isinstance(tc, dict) and tc.get("id"))
    #                     tf      = getattr(tc, "function", None) or (isinstance(tc, dict) and tc.get("function") or {})
    #                     tc_name = (getattr(tc, "name", None)
    #                             or (isinstance(tc, dict) and tc.get("name"))
    #                             or (isinstance(tf, dict) and tf.get("name")))
    #                     tc_args = (getattr(tc, "args", None)
    #                             or (isinstance(tc, dict) and tc.get("args"))
    #                             or (isinstance(tf, dict) and tf.get("arguments")))

    #                     print("\n\n---TOOL (function-calling) CONSOLIDATED---")
    #                     print(f"Tool: {tc_name} (id: {tc_id})")
    #                     print("Args:", tc_args)
    #             else:
    #                 # Si no tenemos tool_calls consolidados, usa el acumulador SOLO si hay datos reales
    #                 for tc_id, meta in tool_calls_meta.items():
    #                     raw = tool_calls_args.get(tc_id, "")
    #                     if not raw and not meta.get("name"):
    #                         continue  # evita imprimir Tool: None (id: None)
    #                     print("\n\n---TOOL (function-calling) END---")
    #                     print(f"Tool: {meta.get('name')} (id: {tc_id})")
    #                     print("Args (raw JSON):", raw)

    #             print("\n\n---CHAT MODEL RESPONSE (completa)---")
    #             print(output)

    #         # ---------- TOOLS clásicas (si tu pipeline emite estos eventos) ----------
    #         if etype == "on_tool_start":
    #             print("\n---TOOL START---")
    #             # En algunos casos 'serialized' no está; probamos varios campos
    #             tool_name = (data.get("name")
    #                         or data.get("tool")
    #                         or (data.get("serialized") or ["?"])[-1])
    #             print("Tool:", tool_name)
    #             print("Input:", data.get("input_str") or data.get("tool_input") or data.get("input"))

    #         if etype == "on_tool_end":
    #             print("\n---TOOL END---")
    #             print("Output:", data.get("output"))

    #         # ---------- Acciones del agente (LangChain Agent) ----------
    #         if etype == "on_agent_action":
    #             # Suelen venir objetos ToolAgentAction
    #             action = data.get("output") or data.get("chunk")
    #             print("\n---AGENT ACTION---")
    #             print(action)

    #         # A veces las acciones llegan en streams de cadena
    #         if etype in ("on_chain_stream", "on_chain_end"):
    #             # Busca ToolAgentAction(s) embebidos
    #             for key in ("chunk", "output"):
    #                 val = data.get(key)
    #                 if not val:
    #                     continue
    #                 # Puede ser lista de acciones
    #                 if isinstance(val, list):
    #                     for item in val:
    #                         if hasattr(item, "tool") and hasattr(item, "tool_input"):
    #                             print("\n---AGENT ACTION (chain)---")
    #                             print(f"Tool: {item.tool}")
    #                             print(f"Input: {item.tool_input}")
    #                 # O dict con 'actions'
    #                 if isinstance(val, dict) and "actions" in val:
    #                     for item in val["actions"]:
    #                         if hasattr(item, "tool") and hasattr(item, "tool_input"):
    #                             print("\n---AGENT ACTION (chain)---")
    #                             print(f"Tool: {item.tool}")
    #                             print(f"Input: {item.tool_input}")

    #         # ---------- Mensajes de tool (resultado) ----------
    #         # Muchos pipelines meten el resultado como ToolMessage en el "scratchpad" o messages
    #         if etype in ("on_chain_stream", "on_chain_end", "on_chat_model_stream", "on_chat_model_end"):
    #             for key in ("agent_scratchpad", "messages"):
    #                 seq = data.get(key)
    #                 if not seq:
    #                     continue
    #                 try:
    #                     for m in seq:
    #                         if isinstance(m, ToolMessage):
    #                             print("\n---TOOL RESULT (ToolMessage)---")
    #                             print(m.content)
    #                 except Exception:
    #                     # Si no tenemos la clase ToolMessage, intentamos detectar por forma
    #                     try:
    #                         for m in seq:
    #                             if getattr(m, "__class__", type("X",(object,),{})).__name__ == "ToolMessage":
    #                                 print("\n---TOOL RESULT (ToolMessage-like)---")
    #                                 print(getattr(m, "content", m))
    #                     except Exception:
    #                         pass

    #         # ---------- Fin del agente ----------
    #         if etype == "on_agent_finish":
    #             print("\n---AGENT FINISH---")
    #             print(data)
    # asyncio.run(stream_agent_response())
