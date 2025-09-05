import logging
from langchain_neo4j import Neo4jGraph
from typing import get_args
import json
from config.common_settings import settings
from langchain_openai import OpenAIEmbeddings
# Set up a logger for the chain
logger = logging.getLogger(__name__)



def connect_to_neo4j():

    #connect to neo4j
    try:
        graph = Neo4jGraph(
            url=settings.NEO4J_URI_BOLT, 
            username=settings.NEO4J_USER, 
            password=settings.NEO4J_PASSWORD, 
            database = settings.NEO4J_DATABASE
        )
        logger.info("Connected to Neo4j successfully")
        return graph

    except Exception as e:
        logger.info(f"Error while connecting to Neo4j: {e}")
        return

def build_representative_text_from_node_properties(graph):
    logger.info("Strating to unify the properties of a node into a single text field...")
    
    # personas
    logger.info("Generating representative text for PERSONAS nodes...")
    graph.query("""
    MATCH (p:Persona)
    SET p.text = 'Persona: ' + coalesce(p.nombre, '') +
                ', tipo: ' + coalesce(p.tipo, '') +
                ', descripción: ' + coalesce(p.descripcion, '') +
                ', profesión: ' + coalesce(p.profesion, '')
    """)

    #empresas
    logger.info("Generating representative text for EMPRESAS nodes...")
    graph.query("""
    MATCH (e:Empresa)
    SET e.text = 'Empresa: ' + coalesce(e.nombre, '') +
                ', tipo: ' + coalesce(e.tipo, '') +
                ', descripción: ' + coalesce(e.descripcion, '') +
                ', industria: ' + coalesce(e.industria, '')
    """)

    #centros educativos
    logger.info("Generating representative text for CENTROS EDUCATIVOS nodes...")
    graph.query("""
    MATCH (c:centroeducativo)
    SET c.text = 'Centro Educativo: ' + coalesce(c.nombre, '') +
                ', tipo: ' + coalesce(c.tipo, '') +
                ', descripción: ' + coalesce(c.descripcion, '') +
                ', localización: ' + coalesce(c.localizacion, '')
    """)

    #movimientos
    logger.info("Generating representative text for MOVIMIENTOS nodes...")
    graph.query("""
    MATCH (m:movimiento)
    SET m.text = 'Movimiento: ' + coalesce(m.nombre, '') +
                ', tipo: ' + coalesce(m.tipo, '') +
                ', descripción: ' + coalesce(m.descripcion, '') +
                ', categoría: ' + coalesce(m.categoria, '')
    """)

    #productos
    logger.info("Generating representative text for PRODUCTOS nodes...")
    graph.query("""
    MATCH (p:producto)
    SET p.text = 'Producto: ' + coalesce(p.nombre, '') +
                ', tipo: ' + coalesce(p.tipo, '') +
                ', descripción: ' + coalesce(p.descripcion, '') +
                ', subtipo: ' + coalesce(p.subtipo, '')
    """)

def build_representative_text_for_relationships(graph):
    logger.info("Starting to unify the properties of a relationship into a single text field...")
    logger.info("Generating representative text for PRODUCTOS nodes...")
    graph.query("""
    MATCH ()-[r:RELACION]->()
    SET r.text = 
    'Relación: ' + coalesce(r.descripcion_relacion, '') +
    ', fuerza: ' + toString(coalesce(r.fuerza_relacion, '')) +
    ', id: ' + coalesce(r.id, '')
    """)

def convert_relationships_as_nodes(graph):
    logger.info("Materializing relationships as nodes with combined text...")
    graph.query("""
    MATCH (origen)-[r:RELACION]->(destino)
    WHERE origen.text IS NOT NULL AND destino.text IS NOT NULL AND r.text IS NOT NULL
    MERGE (relMat:RelMaterializada {id: r.id})
    SET relMat.text = 
    'Entidad de origen: ' + origen.text + '\n' +
    'Entidad de destino: ' + destino.text + '\n' +
    'Relación: ' + r.text
    WITH origen, destino, relMat
    MERGE (relMat)-[:FROM]->(origen)
    MERGE (relMat)-[:TO]->(destino);

    """)

def set_common_label(graph):
    """
    Set a common label for all entities in the graph named "Entity"
    to build a unique index for the graph.
    This is done to avoid having to create a unique index for each entity type.
    """

    graph.query("""
    MATCH (n)
    WHERE any(lbl IN labels(n) WHERE lbl IN [
    'Persona','Empresa','producto','movimiento','centroeducativo','RelMaterializada'
    ])
    SET n:Entity;
    """)


def generate_embeddings(graph):
    """
    Generate embeddings for all nodes in the graph that don't have an embedding yet.
    The embeddings are then set as a property of the node.

    """
    emb = OpenAIEmbeddings(model=settings.EMBEDDINGS_MODEL,
                           api_key=settings.OPENAI_API_KEY)

    # search for nodes that don't have an embedding yet
    records = graph.query(
        """
        MATCH (n:Entity)
        WHERE n.text IS NOT NULL AND n.embedding IS NULL
        RETURN elementId(n) AS eid, n.text AS text
        """
    )
    #set embeddings as a property of the node
    for rec in records:
        node_id = rec["id"]  #get id property
        node_text = rec["text"] #get text property
        vector = emb.embed_query(node_text)  #generate embeddings
        graph.query(
            """
            MATCH (n) WHERE elementId(n) = $eid
            SET n.embedding = $vector
            """,
            params = {
                "id": node_id, 
                "vector": vector
            }
        )

def create_index(graph):
    """
    Create vector index over embeddings for all entities (label Entity)
    """
    graph.query("""
    CREATE VECTOR INDEX entity_emb IF NOT EXISTS
    FOR (n:Entity) ON (n.embedding)
    OPTIONS {
                indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
    }};
    """)

def prepare_graph_embeddings_index():
    logger.info("Trying to connect to Neo4j...")
    graph = connect_to_neo4j()

    logger.info("Generating a unified text using all node properties...")
    build_representative_text_from_node_properties(graph)

    logger.info("Generating a unified text using all relationship properties...")
    build_representative_text_for_relationships(graph)
    
    logger.info("Materializing relationships as nodes with combined text...")
    convert_relationships_as_nodes(graph)

    logger.info("Setting a common label for all nodes for indexing...")
    set_common_label(graph)

    logger.info("Generating embeddings for all nodes...")
    generate_embeddings(graph)

    logger.info("Creating a vector index over embeddings...")
    create_index(graph)


prepare_graph_embeddings_index()