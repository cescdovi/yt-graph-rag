import logging
from typing import Dict, List
from pathlib import Path
from pydantic import PrivateAttr, ConfigDict
from langchain.chains.base import Chain
from langchain_core.runnables import RunnableSequence
import os
import json

# Set up a logger for the chain
logger = logging.getLogger(__name__)

class CorreferenceResolutionChain(Chain):
    """ 
    Chain that performs coreference resolution on a unified text.

    Purpose
    -------
    - Takes a `video_id` and a corrected transcript.
    - Extracts video metadata (title and description) to provide context for the LLM.
    - Uses a `correference_resolution_chain` to perform the coreference resolution.
    - Saves the final text to a file.
    - Returns the `video_id` and the coreference-resolved text.

    Attributes
    ----------
    input_keys : List[str]
        Expected input keys: `["_video_id", "corrected_text"]`.
    output_keys : List[str]
        Returned output keys: `["_video_id", "correference_resolution_text"]`.
    """

    _correference_resolution_chain: RunnableSequence = PrivateAttr()
    model_config = ConfigDict(extra="ignore") # ignore unexpected fields in input dictionary

    @property
    def input_keys(self) -> List[str]:
        return ["_video_id", "corrected_text"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["_video_id", "correference_resolution_text"]
    
    def __init__(self,
            correference_resolution_chain: RunnableSequence, 
            **kwargs):
        super().__init__(**kwargs)
        self._correference_resolution_chain = correference_resolution_chain
        logger.info("CorreferenceResolutionChain initialized.")

    def _call(self, inputs: Dict) -> Dict:
        _video_id = inputs["_video_id"]
        text_to_correct_correferences = inputs.get("corrected_text")

        if not text_to_correct_correferences:
            logger.warning("No corrected text provided for video_id=%s. Returning empty result.", _video_id)
            return {
                "_video_id": _video_id,
                "correference_resolution_text": ""
            }

        logger.info(
            "Starting coreference resolution for video_id=%s. Text length: %d",
            _video_id,
            len(text_to_correct_correferences)
        )

        video_title = ""
        video_description = ""
        
        # Extract metadata as context for coreference resolution
        try:
            from config.common_settings import settings
            metadata_path = Path(settings.DATA_DIR) / _video_id / "metadata.json"
            if not metadata_path.exists():
                logger.warning("Metadata file not found at %s. Proceeding without context.", metadata_path)
            else:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata_json = json.load(f)
    
                video_title = metadata_json.get(_video_id, {}).get("title", "")
                video_description = metadata_json.get(_video_id, {}).get("description", "")
                logger.info("Metadata loaded successfully. Title: '%s', Description: '%s'", video_title, video_description)
        
        except ImportError:
            logger.error("Could not import `settings`. Skipping metadata extraction.")
        except Exception as e:
            logger.error(
                "Error extracting metadata for video_id=%s: %s",
                _video_id,
                e,
                exc_info=True
            )

        # Perform coreference resolution
        try:
            result = self._correference_resolution_chain.invoke({
                "video_title": video_title,
                "video_description": video_description,
                "text_to_correct_correferences": text_to_correct_correferences
            })
            correference_resolution_text = result.content
            logger.debug("Coreference resolution completed. Result length: %d", len(correference_resolution_text))
        except Exception as e:
            logger.error(
                "Error during coreference resolution for video_id=%s: %s",
                _video_id,
                e,
                exc_info=True
            )
            raise

        # Save the result to a file
        try:
            from config.common_settings import settings
            output_dir = Path(settings.DATA_DIR) / _video_id / "texts" / "correference_resolution"
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"correference_resolution_{_video_id}.txt"
            file_path = output_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(correference_resolution_text)
            
            logger.info("Coreference-resolved text saved to file: %s", file_path)

        except ImportError:
            logger.error("Could not import `settings`. Skipping file save.")
        except Exception as e:
            logger.error(
                "Error saving coreference-resolved text for video_id=%s: %s",
                _video_id,
                e,
                exc_info=True
            )
        
        logger.info("Coreference resolution chain finished for video_id=%s.", _video_id)
                
        return {
            "_video_id": _video_id,
            "correference_resolution_text": correference_resolution_text
            }
