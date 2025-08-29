import logging
from typing import Dict, List
from pathlib import Path
from pydantic import PrivateAttr, ConfigDict
from langchain.chains.base import Chain
from langchain_core.runnables import RunnableSequence

# Set up a logger for the chain
logger = logging.getLogger(__name__)

class UnifyTranscriptsChain(Chain):
    """ 
    Chain that unifies the transcripts from a YouTube video's chunks.

    Purpose
    -------
    - Takes a `video_id` and a list of chunk transcripts.
    - Uses an internal `unifier_chain` to progressively merge the transcripts.
    - Saves the unified transcript to a text file.
    - Returns the `video_id` and the unified transcript.

    Attributes
    ----------
    input_keys : List[str]
        Expected input keys: `["_video_id", "transcripts"]`.
    output_keys : List[str]
        Returned output keys: `["_video_id", "unified_transcript"]`.
    """

    # Use ConfigDict to ignore fields not defined in the class
    model_config = ConfigDict(extra="ignore")

    # Use PrivateAttr so the internal chain is not part of the Pydantic model
    _unifier_chain: RunnableSequence = PrivateAttr()

    @property
    def input_keys(self) -> List[str]:
        return ["_video_id", "transcripts"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["_video_id", "unified_transcript"]
    
    def __init__(self,
                 unifier_chain: RunnableSequence,
                 **kwargs
                 ):
        """
        Initializes the chain with the RunnableSequence that will perform the unification.
        """
        super().__init__(**kwargs)
        self._unifier_chain = unifier_chain
        logger.info("UnifyTranscriptsChain initialized.")

    def _call(self, inputs: Dict) -> Dict:
        """
        The main method that executes the chain's logic.
        """
        _video_id = inputs['_video_id']
        chunks_transcripted = inputs.get("transcripts", [])
        
        logger.info(
            "Starting transcript unification for video_id=%s. Number of chunks: %d",
            _video_id,
            len(chunks_transcripted)
        )

        if not chunks_transcripted:
            logger.warning("No transcripts found for video_id=%s. Returning empty result.", _video_id)
            return {
                "_video_id": _video_id,
                "unified_transcript": ""
            }

        unified_transcript: str = ""

        for idx, chunk in enumerate(chunks_transcripted):
            logger.debug("Processing chunk %d for video_id=%s", idx, _video_id)
            try:
                result = self._unifier_chain.invoke({
                    "unified_text": unified_transcript,
                    "chunk_text": chunk
                })
                unified_transcript = result.content
                logger.debug(
                    "Chunk %d processed. Current unified transcript length: %d",
                    idx,
                    len(unified_transcript)
                )
            except Exception as e:
                logger.error(
                    "Error processing chunk %d for video_id=%s: %s",
                    idx,
                    _video_id,
                    e,
                    exc_info=True
                )
                raise
        
        # NOTE: `settings` is not a standard Python import. Assuming it's defined elsewhere.
        # This part assumes a valid `settings.DATA_DIR` exists.
        try:
            from config.common_settings import settings
            text_dir = Path(settings.DATA_DIR) / _video_id / "texts" / "unified_chunks"
            text_dir.mkdir(parents=True, exist_ok=True)
            filename = f"unified_{_video_id}.txt"
            file_path = text_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(unified_transcript)
            
            logger.info("Unified transcript saved to file: %s", file_path)

        except ImportError:
            logger.error("Could not import `settings`. Skipping file save.")
            
        except Exception as e:
            logger.error(
                "Error saving unified transcript for video_id=%s: %s",
                _video_id,
                e,
                exc_info=True
            )
        
        logger.info("Transcript unification completed for video_id=%s. Final length: %d",
                    _video_id,
                    len(unified_transcript)
        )
                
        return {
            "_video_id": _video_id,
            "unified_transcript": unified_transcript
        }
