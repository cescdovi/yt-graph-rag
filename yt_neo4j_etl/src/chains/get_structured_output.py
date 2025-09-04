import logging
from typing import Dict, List
from pathlib import Path
from pydantic import PrivateAttr, ConfigDict
from langchain.chains.base import Chain
from langchain_core.runnables import RunnableSequence

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
