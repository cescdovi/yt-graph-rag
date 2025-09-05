import logging
from langchain_neo4j import Neo4jGraph
from typing import get_args
import json
from config.common_settings import settings
from yt_neo4j_etl.src.pydantic_models.pydantic_models import OutputSchema
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
        print(f"Error while connecting to Neo4j: {e}")
        return

def _set_uniqueness_constraints(graph, node):
    """Set a uniqueness constraint on the given node type."""

    graph.query(f"""
    CREATE CONSTRAINT IF NOT EXISTS
    FOR (n:{node}) REQUIRE n.id IS UNIQUE;
    """)


def etl_load_to_neo4j(inputs: dict):
    logger.info("Trying to connect to Neo4j...")
    graph = connect_to_neo4j()

    input_str = inputs["structured_output"]
    input_json = json.loads(input_str)
    obj_validated = OutputSchema.model_validate(input_json)

    #extract entities
    ENTITIES = [
    get_args(tipo)[0].__name__  # extract type in List[Type]
    for tipo in obj_validated.entidades.__annotations__.values()
    ]

    #set constraints for each node type
    logger.info("Setting uniqueness constraints on nodes")
    for entity in ENTITIES:
        _set_uniqueness_constraints(graph, entity)

    
    #load data
    logger.info("Strating to load nodes to Neo4j...")
    logger.info("Loading PERSONAS to Neo4j...")
    for persona in obj_validated.entidades.personas:
        graph.query(
            """
            MERGE (p:Persona {id: $id})
            SET p.tipo = $tipo, p.nombre = $nombre, p.descripcion = $descripcion, p.profesion = $profesion
            """,
            params = {
                "id": persona.id,
                "tipo": persona.tipo,
                "nombre": persona.nombre,
                "descripcion": persona.descripcion,
                "profesion": persona.profesion,
    
            }

        )

    logger.info("Loading EMPRESAS to Neo4j...")
    for empresa in obj_validated.entidades.empresas:
        graph.query(
            """
            MERGE (e:Empresa {id: $id})
            SET e.tipo = $tipo, e.nombre = $nombre, e.descripcion = $descripcion, e.industria = $industria
            """,
            params = {
                "id": empresa.id,
                "tipo": empresa.tipo,
                "nombre": empresa.nombre,
                "descripcion": empresa.descripcion,
                "industria": empresa.industria,
    
            }

        )
    logger.info("Loading CENTROSEDUCATIVOS to Neo4j...")
    for centro in obj_validated.entidades.centros_educativos:
        graph.query(
            """
            MERGE (c:centroeducativo {id: $id})
            SET c.tipo = $tipo, c.nombre = $nombre, c.descripcion = $descripcion, c.localizacion = $localizacion
            """,
            params = {
                "id": centro.id,
                "tipo": centro.tipo,
                "nombre": centro.nombre,
                "descripcion": centro.descripcion,
                "localizacion": centro.localizacion,
    
            }

        )
    logger.info("Loading MOVIMIENTOS to Neo4j...")
    for movimiento in obj_validated.entidades.movimientos:
        graph.query(
            """
            MERGE (m:movimiento {id: $id})
            SET m.tipo = $tipo, m.nombre = $nombre, m.descripcion = $descripcion, m.categoria = $categoria
            """,
            params = {
                "id": movimiento.id,
                "tipo": movimiento.tipo,
                "nombre": movimiento.nombre,
                "descripcion": movimiento.descripcion,
                "categoria": movimiento.categoria,
    
            }

        )

    logger.info("Loading PRODUCTOS to Neo4j...")
    for producto in obj_validated.entidades.productos:
        graph.query(
            """
            MERGE (p:producto {id: $id})
            SET p.tipo = $tipo, p.nombre = $nombre, p.descripcion = $descripcion, p.subtipo = $subtipo
            """,
            params = {
                "id": producto.id,
                "tipo": producto.tipo,
                "nombre": producto.nombre,
                "descripcion": producto.descripcion,
                "subtipo": producto.subtipo,
    
            }

        )
    logger.info("Loading RELACIONES to Neo4j...")
    for relacion in obj_validated.relaciones.relaciones:
        graph.query(
            """
            MATCH (origen {id: $id_origen})
            MATCH (destino {id: $id_destino})
            MERGE (origen)-[r:RELACION]->(destino)
            SET r.descripcion_relacion = $descripcion_relacion,
                r.fuerza_relacion = $fuerza_relacion,
                r.id = $id_relacion
            """,
            params={
                "id_origen": relacion.entidad_origen.id,
                "id_destino": relacion.entidad_destino.id,
                "descripcion_relacion": relacion.descripcion_relacion,
                "fuerza_relacion": relacion.fuerza_relacion,
                "id_relacion": relacion.id,
            }
        )

# graph = connect_to_neo4j()
# graph.query("MATCH (n) DETACH DELETE n")