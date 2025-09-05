from langchain_redis import RedisChatMessageHistory
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
    streaming=True,
    temperature=0,
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

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
    return result["result"]


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
system_prompt = (
    "Eres un agente útil. "
    "Cuando la pregunta requiera información actual o fuentes, usa la tool de búsqueda web. "
    "Cuando se pida información del grafo, razona y luego usa la tool neo4j_query con consultas READ (MATCH/RETURN). "
    "Devuelve respuestas concisas y con pasos si ejecutaste tools."
)
human_prompt = "{input}"

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# brain that decides what to do
agent_runnable = create_tool_calling_agent(
    llm,
    tools,
    prompt,
    # handle_parsing_errors=True,  # opcional para forzar uso de tools
)

# engine that  execute tools and manage the 
AGENT = AgentExecutor(agent=agent_runnable, tools=tools, verbose=True)

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


    # import asyncio
    # q1 = "En que empresa trabajó Silvia García?"
    # chat_model_response_printed = False
    # async def stream_agent_response():
    #     chat_model_response_printed = False

    #     async for ev in agent.astream_events({"input": q1}):
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
    # asyncio.run(stream_agent_response())
