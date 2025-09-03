import logging
from typing import Dict, List
from pathlib import Path
from pydantic import PrivateAttr, ConfigDict
from langchain.chains.base import Chain
from langchain_core.runnables import RunnableSequence
from yt_neo4j_etl.src.pydantic_models.pydantic_models import OutputSchema

# Set up a logger for the chain
logger = logging.getLogger(__name__)


class GetStructuredOutputChain(Chain):
    """Chain to get structured output from a chain."""

    _structured_output_chain: RunnableSequence = PrivateAttr()
    model_config = ConfigDict(extra="ignore") # ignore unexpected fields in input dictionary

    @property
    def input_keys(self) -> List[str]:
        return ["_video_id", "spanish_text"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["_video_id", "structured_output"]
    
    def __init__(self,
            structured_output_chain: RunnableSequence, 
            **kwargs):
        super().__init__(**kwargs)
        self._structured_output_chain = structured_output_chain
        logger.info("GetStructuredOutputChain initialized.")

    def _call(self, inputs: List[Dict]) -> List[Dict]:
        _video_id = inputs["_video_id"]
        spanish_text = inputs.get("spanish_text")

        if not spanish_text:
            logger.warning("No plain text provided for video_id=%s. Returning empty result.", _video_id)
            return {
                "_video_id": _video_id,
                "spanish_text": ""
            }

        logger.info(
            "Starting structured output generation for video_id=%s.",
            _video_id
        )


        # get structured output from plain text
        try:
            result = self._structured_output_chain.invoke({
                    "text_to_extract_entities": spanish_text
                })
            structured_output = result.model_dump_json(indent=2)
            logger.debug(
                "Structured output  completed for video: %s",
                _video_id
            )
        except Exception as e:
            logger.error(
                "Error during correction for video_id=%s: %s",
                _video_id,
                e,
                exc_info=True
            )
            raise
        

        

        #load structured output to a json file
        try:
            from config.common_settings import settings
            json_dir = Path(settings.DATA_DIR) /_video_id / "texts" / "structured"
            json_dir.mkdir(parents=True, exist_ok=True)
            filename = f"structured_{_video_id}.json"
            file_path = json_dir / filename

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(structured_output)

            logger.info("Corrected structured output saved to file: %s", file_path)
       
        except ImportError:
            logger.error("Could not import `settings`. Skipping file save.")
            
        except Exception as e:
            logger.error(
                "Error saving corrected transcript for video_id=%s: %s",
                _video_id,
                e,
                exc_info=True
            )
        
        logger.info("Structured output generation finished for video_id=%s.", _video_id)
                
        return {
            "_video_id": _video_id,
            "structured_output": structured_output
            }



        
        




######################################################  
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from config.common_settings import settings
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate


parser = PydanticOutputParser(pydantic_object=OutputSchema)

#src/prompts/prompt_structured_output.py
chat_prompt_structured_outputs = ChatPromptTemplate.from_messages(
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


#src/load_to_ne4j.py
structured_output_parser = PydanticOutputParser(pydantic_object=OutputSchema)

entity_extraction_llm = ChatOpenAI(
    model = "gpt-4.1-2025-04-14",
    api_key = settings.OPENAI_API_KEY,
    max_retries = 2
)
#chain = prompt | entity_extraction_llm | parser

get_structured_output_chain = GetStructuredOutputChain(
    structured_output_chain=(chat_prompt_structured_outputs.partial(format_instructions=parser.get_format_instructions()) | entity_extraction_llm | parser))


result_translation = {'_video_id': 'BQFmYHFieqU',
    'spanish_text': 'Yo, Silvia García, vengo de una familia en la que la creatividad y la estética siempre han tenido mucha importancia, pero siempre ligada a la cultura del esfuerzo y el coraje. Silvia García habla de estética no en el sentido frívolo de la palabra, sino en el sentido del trabajo bien hecho, de cuidar los detalles. El concepto de estética es algo que Silvia García siempre ha vivido, sobre todo en la familia paterna de Silvia García.\n\nCuando Silvia García terminó tercero de carrera, Silvia García se casó con su marido, Rogelio, que trabajaba en la Mediterránea. Rogelio era el director financiero de la Mediterránea. Rogelio le comentó a Silvia García que el gerente de la Mediterránea estaba pensando en abrir un apartado dentro de Mediterránea dedicado a la decoración del vidrio. Silvia García fue a hablar con el gerente de la Mediterránea y el gerente le dio la oportunidad de empezar ese camino y ahí Silvia García empezó a trabajar en la Mediterránea.\n\nSilvia García, en ese aspecto, tuvo un aprendizaje muy bueno porque su suegro, el padre de Rogelio, desde pequeño había trabajado en las fábricas de vidrio y tenía una experiencia brutal en ese campo, de hecho el padre de Rogelio era encargado de uno de los turnos de producción de la Mediterránea.\n\nDe aquellos inicios en la Mediterránea, lo que permitió a Silvia García crecer dentro de la empresa fueron dos o tres colecciones que tuvieron mucho éxito. Estas colecciones estaban decoradas con lustre. Silvia García tuvo la suerte de que estas colecciones las cogieron varias empresas, entre ellas una empresa americana, que importaron cantidades enormes de ese producto y, de alguna manera, el éxito de estos productos, aunque ahora puedan parecer desfasados, en aquella época no los hacía nadie y eso le dio a Silvia García el aval para hacer otras cosas dentro de la Mediterránea.\n\nLa Mediterránea nació en el año 75 a raíz de la crisis del petróleo. Hubo una crisis muy fuerte que afectó a las empresas textiles, las empresas del vidrio, todas las empresas que trabajaban con petróleo. El coste energético subió, bajaron los sueldos, las condiciones laborales empeoraron, coincidió con toda la etapa de la Transición, sociológicamente había un sentido revolucionario, la gente empezó a asociarse en sindicatos y todo ese conjunto de factores hizo que en l’Olleria nacieran tres cooperativas prácticamente al mismo tiempo, de gente que salió de las empresas privadas y se unió para crear empresas cooperativas, y uno de esos casos fue la Mediterránea.\n\nEl hecho de que la Mediterránea fuera una cooperativa, para Silvia García ha sido muy importante, era una empresa emblemática en el sentido participativo de todo el mundo, en el sentido de que todo el mundo participaba de la propiedad de la Mediterránea.\n\nLos inicios de la Mediterránea son muy románticos, gente que viene del mundo del vidrio sin formación en empresas privadas de vidrio, que en un momento se reunían clandestinamente, porque todavía no estaba permitido, y crearon una cooperativa. Con el tiempo la Mediterránea se convirtió en una empresa muy potente, es algo muy bonito.\n\nSilvia García estaba encantada, la verdad, por el hecho de que la Mediterránea fuera una cooperativa. Pronto se unió al grupo cooperativo valenciano, y Silvia García cree que fue una influencia muy positiva, porque era una forma de intercambiar experiencias, puntos de vista empresariales, recursos. En la Mediterránea se tuvo muy claro desde el principio la formación interna; después, a nivel social, Mediterránea pagaba los libros de todos los hijos de los trabajadores, las matrículas, temas deportivos o cuestiones sociales, se realizaban muchas, una gran parte de los beneficios de Mediterránea se dedicaba a este tipo de cuestiones.\n\nLa relación que el Departamento de Diseño de la Mediterránea tenía con las otras áreas de la empresa varió mucho a lo largo del tiempo. En los primeros años, como los primeros diez años, Silvia García tenía una posición de dependencia de gerencia, y el contacto que tenía con el Departamento Comercial era un contacto informativo, eran compañeros, pero no de dependencia. La persona con la que despachaba era con el gerente. Silvia García recuerda esa etapa como un crecimiento muy fluido.\n\nDespués, a lo largo de la historia de la Mediterránea, hubo distintas etapas, por ejemplo la etapa en la que se implantó la ISO, en la que los asesores que llevaban la implantación de la ISO reorganizaron todo el organigrama de la empresa y situaron el Departamento de Diseño por debajo de la Dirección Comercial. Cambió mucho la situación porque Silvia García ya no tenía la misma libertad para hacer cosas, ni las propuestas que hacía llegaban de la misma manera a la gerencia, siempre estaban filtradas por el Departamento Comercial.\n\nLa aportación de Silvia García a la Mediterránea, Silvia García cree que es sobre todo una forma diferente de ver los proyectos, una forma, si se quiere, un poco de arriesgar y de cuestionar lo obvio y plantearlo de otra forma, darle la vuelta, un poco el pensamiento filosófico de cuestionarlo todo y de situarse un poco fuera y mirar la situación desde otra perspectiva a ver si se puede hacer de otra manera. Esta actitud ha llevado a Silvia García a veces a enfrentamientos con el responsable de producción, con el encargado de decoración, que decían que eso no se podía hacer, que no era rentable, que no era productivo, pero no es verdad, muchas veces puedes equivocarte pero muchas veces al darle la vuelta se encuentran caminos diferentes que llevan a diferenciarse y a crecer.\n\nEl hecho de ser mujer no facilitó las cosas a Silvia García, evidentemente, Silvia García tuvo que demostrar tres veces más que podía hacer ese trabajo y que podía hacerlo bien. No fue fácil, realmente, en la fábrica, las únicas chicas que había en la Mediterránea eran las de la Arca, que eran las chicas que envolvían. Al salir el producto del Arca, una vez ya estaba frío, ese trabajo normalmente lo hacían las chicas, pero lo que es la producción en sí, todos los cargos de responsabilidad, los encargados, los jefes de taller, los directores de producción, todos eran hombres, y sigue siendo así, y de vidrieras no hay, ninguna mujer vidriera.\n\nEl IMPIVA fue un organismo que Silvia García cree que también fue muy importante, en el sentido de que el IMPIVA avaló todo el desarrollo de exposición en ferias de la Mediterránea, es decir, el IMPIVA dio ayudas importantes a todo el proceso de ferias, ahí sí, al exterior, que fue lo que hizo que Mediterránea realmente despegara a nivel de exportación. Mediterránea llegó a tener un 85% de sus ventas en exportación.\n\nLas colecciones que más repercusión han tenido para Silvia García a nivel personal son las colecciones que han abierto nuevos caminos dentro de la Mediterránea. Silvia García recuerda sobre todo una colección, una vajilla, que es la Amare, porque tenía motivos marinos, y Silvia García la recuerda mucho porque fue un éxito comercial muy importante, y porque abrió toda un área de vajillas en la Mediterránea que hasta el momento no existía, porque requería de una tecnología que hasta entonces no había en la Mediterránea, se necesitaban máquinas centrífugas para poder fabricar los platos.\n\nDespués, Silvia García recuerda otra vajilla a la que también tiene mucho cariño, que es una vajilla que hicieron en vidrio plano, en vidrio de fusión, que son planchas de vidrio de ventana, que se doblaban sobre unos moldes, era una técnica totalmente diferente a la centrífuga, y que luego, como eran piezas muy planas, se decoraban en serigrafía. El aspecto estético de la colección era como si fueran hojas de periódico con artículos. Así que la idea era que el diseño de los platos, con los vasos, con las jarras, tuvieran notas de prensa utópicas. Una, por ejemplo, hacía referencia al mundo de la infancia, otra sobre las guerras, otra sobre la ecología.\n\nDespués, otra colección que se desarrolló y que tuvo mucho éxito porque la compraron las tiendas del MoMA fue la colección Aster, que era una colección que también nació de forma muy libre, era una colección de platos en una gama muy amplia de colores, y la idea era que una vajilla no fuera toda del mismo color, sino que cada persona pudiera componer su vajilla en los colores que más le gustasen.\n\nClaro, hay tantísimos productos a lo largo de tantos años de la Mediterránea que es difícil acotarlo. Una pieza que fue superimportante en la Mediterránea, también por la repercusión mediática que tuvo, fue la edición de la Siesta, que fue un contacto que tuvo Alberto Martínez un día y Alberto Martínez presentó el producto a Silvia García. Alberto Martínez le dijo que era un producto que habían hecho en la escuela, en Londres, y que lo había hecho junto a Héctor Serrano y Enrique Martínez. Entonces, Alberto Martínez le explicaba a Silvia García que él, Héctor Serrano y Enrique Martínez lo habían hecho y tanto a Silvia García como al equipo de la Mediterránea les pareció que era un producto perfecto para la Mediterránea. Y la verdad es que Silvia García vio la pieza y se enamoró de ella enseguida. La pieza tenía una carga simbólica que era ideal para que la Mediterránea la produjera. La Mediterránea tenía por estrategia que el producto debía ser de la Mediterránea. Pero sí que es verdad que a lo largo de su historia diferentes empresas diseñadoras se han ido acercando y en algunos momentos ha sido posible llevar a cabo esa producción.\n\nLa primera, Silvia García cree que fue la de Sybilla, la diseñadora de moda, que hicieron una colección de tulipanes en vidrio de masa de color y la base de cerámica y después las varillas también eran preciosas. Las varillas también estaban diseñadas, tenían una sensibilidad, eran muy bonitas la colección de Sybilla. Y después también hicieron otra colección que eran unos cuencos que se unían entre sí, también de cerámica. Esta es la primera colaboración. Después, a raíz de esta colaboración, Ágatha Ruiz de la Prada vio la colaboración con Sybilla y dijo que la Mediterránea era una empresa que trabajaba mucho el color, que trabajaba un producto 100% reciclado, y propuso también hacer la colaboración con ella.\n\nDesde el punto de vista de Silvia García, las causas del cierre de la Mediterránea cooperativa en 2010 fueron varias. En primer lugar, la Mediterránea llevaba ya unos años sufriendo una crisis brutal en todos los sectores productivos causada por la crisis financiera de 2008 y que afectó terriblemente a la exportación. La Mediterránea tenía todo su potencial en la exportación, así que la exportación cayó en picado.\n\nLa crisis de la Mediterránea se unió a una crisis interna de dos posturas de alguna manera enfrentadas, dos filosofías de hacer empresa diferentes, dos estrategias. Una era la estrategia que apostaba por el valor y la otra por el volumen. Entonces no lo tenían claro. Esta fue una de las razones por la que Silvia García se fue de la Mediterránea, porque no lo tenían claro, y todo eso se juntó con un momento financiero delicado para la empresa porque no hacía mucho tiempo que se había hecho una inversión fortísima en un horno mecanizado de una capacidad de toneladas de vidrio exagerada y la inversión llegó en el peor momento. También se sumaba el hecho de que había un alto número de trabajadores. En ese momento había 430 trabajadores en la Mediterránea. Entonces era muy difícil deshacer eso, era muy difícil en una cooperativa saber quién sobra, por lo tanto la flexibilidad era nula. Si se apagaba el horno no se podía volver a poner en marcha, o era muy costoso volverlo a arrancar. La industria del vidrio tiene una serie de características que hacen que sea muy poco flexible pararla. Así que era toda una estructura que para Silvia García estaba orientada al crecimiento continuo y a veces el crecimiento continuo no es posible. Llega un momento en que una crisis como la de 2008 es tan sumamente fuerte que hace decrecer las empresas. Pero todo estaba orientado a crecer, el horno gigante, la cantidad exagerada de mano de obra directa, muy difícil de deshacer.\n\nEn la segunda etapa de la Mediterránea, cuando esta era propiedad de capital riesgo, la experiencia que Silvia García tuvo fue completamente diferente a la de la cooperativa. Por las formas de trabajar, por las formas de entender la Mediterránea, por todo, por la sensibilidad al proceso en sí, al proceso de fabricación. En esta etapa fue cuando las primeras colecciones que se produjeron fueron la colección Cocó, la colección Maiaya, que es la que estuvo en el MoMA. Especialmente la colección Maiaya tiene un significado muy importante para Silvia García porque fue un homenaje a su abuela. Y fue una de las colecciones que inmediatamente, cuando se presentaron en la feria de Maison, el MoMA también la seleccionó.\n\nDespués, en este caso, sí que se buscó la colaboración externa y Silvia García buscó a Héctor Serrano en la misión de que Héctor Serrano reinterpretara clásicos del sector del vidrio, como por ejemplo el porrón. Entonces empezaron editando el "porrón pompero", que se hizo superfamoso. Después, lo que ocurrió es que entre los mismos socios Silvia García piensa que hubo discrepancias, hubo malas historias y esas discrepancias hicieron que la Mediterránea no tuviera la viabilidad que podría haber tenido.\n\nSilvia García cree que la Mediterránea fue una empresa emblemática no solo por el producto, porque era una empresa cooperativa, porque trabajaba una materia prima 100% reciclada, que por aquel entonces, ahora es algo muy obvio, pero entonces no estaba tan claro ni se valoraba.\n\nUna vez cerró la Mediterránea por segunda vez en la etapa de Valcapital, ya no quedaban trabajadores, simplemente había dos o tres personas en la Mediterránea, liquidando el producto que quedaba y demás. Silvia García recibió la llamada de Manuel de Cuona y Manuel de Cuona propuso a Silvia García que recogiera toda la memoria de la Mediterránea. Silvia García hizo tres archivos diferentes: el más completo es el archivo que se llevó Manuel de Cuona a la Universidad Politécnica de Valencia, otro segundo archivo que Silvia García entregó al Ayuntamiento de l’Olleria y un archivo menos completo que tiene Silvia García. Silvia García está muy contenta de haber podido hacer eso, la verdad es que se habría perdido todo, como si la Mediterránea no hubiera existido.'}

result_entity_extraction = get_structured_output_chain.invoke({
    "_video_id": result_translation["_video_id"],
    "spanish_text": result_translation["spanish_text"]

    })