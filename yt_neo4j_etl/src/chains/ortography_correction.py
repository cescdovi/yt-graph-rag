import logging
from typing import Dict, List
from pathlib import Path
from pydantic import PrivateAttr, ConfigDict
from langchain.chains.base import Chain
from langchain_core.runnables import RunnableSequence

# Set up a logger for the chain
logger = logging.getLogger(__name__)

class OrtographyCorrectionChain(Chain):
    """
    Chain that corrects errors in a transcript based on an LLM's context and knowledge.

    Purpose
    -------
    - Takes a `video_id` and a unified transcript as input.
    - Uses an internal `corrective_chain` to perform the spelling and grammar correction.
    - Saves the corrected text to a file.
    - Returns the `video_id` and the corrected text.

    Attributes
    ----------
    input_keys : List[str]
        Expected input keys: `["_video_id", "unified_transcript"]`.
    output_keys : List[str]
        Returned output keys: `["_video_id", "corrected_text"]`.
    """

    _corrective_chain: RunnableSequence = PrivateAttr()
    model_config = ConfigDict(extra="ignore") # ignore unexpected fields in input dictionary
    
    @property
    def input_keys(self) -> List[str]:
        return ["_video_id", "unified_transcript"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["_video_id", "corrected_text"]
    
    def __init__(self,
                corrective_chain: RunnableSequence,
                **kwargs
                ):
        super().__init__(**kwargs)
        self._corrective_chain = corrective_chain
        logger.info("OrtographyCorrectionChain initialized.")
   
    def _call(self, inputs: Dict) -> Dict:
        _video_id = inputs["_video_id"]
        unified_transcript = inputs.get("unified_transcript")
        
        if not unified_transcript:
            logger.warning("No unified transcript provided for video_id=%s. Returning empty result.", _video_id)
            return {
                "_video_id": _video_id,
                "corrected_text": ""
            }

        logger.info(
            "Starting ortography correction for video_id=%s. Transcript length: %d",
            _video_id,
            len(unified_transcript)
        )
        
        try:
            result = self._corrective_chain.invoke({
                "text_to_correct": unified_transcript,
            })
            corrected_text = result.content
            logger.debug(
                "Correction completed. Original text length: %d, Corrected text length: %d",
                len(unified_transcript),
                len(corrected_text)
            )
        except Exception as e:
            logger.error(
                "Error during correction for video_id=%s: %s",
                _video_id,
                e,
                exc_info=True
            )
            raise

        # NOTE: `settings` is not a standard Python import. Assuming it's defined elsewhere.
        # This part assumes a valid `settings.DATA_DIR` exists.
        try:
            from config.common_settings import settings
            text_dir = Path(settings.DATA_DIR) / _video_id / "texts" / "corrected"
            text_dir.mkdir(parents=True, exist_ok=True)
            filename = f"corrected_{_video_id}.txt"
            file_path = text_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(corrected_text)
            
            logger.info("Corrected transcript saved to file: %s", file_path)

        except ImportError:
            logger.error("Could not import `settings`. Skipping file save.")
            
        except Exception as e:
            logger.error(
                "Error saving corrected transcript for video_id=%s: %s",
                _video_id,
                e,
                exc_info=True
            )
        
        logger.info("Ortography correction finished for video_id=%s.", _video_id)
                
        return {
            "_video_id": _video_id,
            "corrected_text": corrected_text
            }
