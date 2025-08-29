import logging
from pydantic import Field, ConfigDict
from typing import Dict, List
from pathlib import Path
from langchain_core.document_loaders import Blob
from langchain.chains.base import Chain
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser

# Set up a logger for the chain
logger = logging.getLogger(__name__)

class WhisperTranscriptionChain(Chain):
    """
    Chain that transcribes audio chunks using the LangChain wrapper for OpenAI Whisper.

    Purpose
    -------
    - Takes a `video_id` and a list of audio chunk file paths.
    - For each chunk, creates a `Blob`, sends it to the `OpenAIWhisperParser`,
      and collects the resulting transcription.
    - Returns a dictionary containing the `video_id` and the list of transcripts
      (in the same order as `chunk_paths`).

    Initialization Parameters
    -------------------------
    parser : OpenAIWhisperParser, optional
        LangChain parser wrapping OpenAI Whisper. If not provided, one will
        be created by default (`OpenAIWhisperParser()`).

    Attributes
    ----------
    input_keys : List[str]
        Expected keys in the input dictionary: `["_video_id", "chunk_paths"]`.
    output_keys : List[str]
        Keys returned by `_call`: `["_video_id", "transcripts"]`.
    """

    parser: OpenAIWhisperParser = Field(default = OpenAIWhisperParser())

    model_config = ConfigDict(extra="ignore") # ignore unexpected fields in input dictionary

    @property
    def input_keys(self) -> List[str]:
        return ["_video_id", "chunk_paths"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["_video_id", "transcripts"]
    
    def __init__(self, 
                 parser: OpenAIWhisperParser,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.parser = parser
        logger.info("WhisperTranscriptionChain initialized.")

    def _call(self, inputs: Dict[str, List[str]]) -> Dict[str, List[str]]:
        _video_id = inputs["_video_id"]
        chunk_paths: List[str] = inputs.get("chunk_paths", [])

        logger.info(
            "Starting Whisper transcription for video_id=%s. Number of chunks: %d",
            _video_id,
            len(chunk_paths)
        )
        
        transcripts: List[str] = []

        for idx, path in enumerate(chunk_paths):
            try:
                p = Path(path)
                if not p.exists():
                    logger.error("Chunk not found (%s) for video_id=%s", path, _video_id)
                    raise FileNotFoundError(f"Chunk not found: {path}")

                logger.debug("Reading bytes from chunk %d: %s", idx, path)
                raw = p.read_bytes()                

                logger.debug("Creating Blob for chunk %d", idx)
                blob = Blob(
                    data = raw,
                    path = str(p),
                    metadata = {"chunk_filename": Path(path).name}
                )

                logger.debug("Invoking OpenAIWhisperParser on chunk %d", idx)
                docs = self.parser.parse(blob)

                text = " ".join(d.page_content for d in docs)
                transcripts.append(text)
                
                logger.debug(
                    "Transcription completed for chunk %d (characters=%d)",
                    idx,
                    len(text),
                )
   
            except Exception as e:
                logger.exception(
                    "Error transcribing chunk %d (%s) for video_id=%s: %s",
                    idx,
                    path,
                    _video_id,
                    e,
                )
                raise

        logger.info(
            "Whisper transcription finished for video_id=%s. Processed chunks: %d", 
            _video_id, 
            len(transcripts)
        )
    
        return {
            "_video_id": _video_id,
            "transcripts": transcripts
        }
