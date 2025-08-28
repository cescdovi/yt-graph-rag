import logging
from pydantic import Field, ConfigDict
from typing import Dict, List
from pathlib import Path
from pydub import AudioSegment
from langchain.chains.base import Chain
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

from config.common_settings import settings

logger = logging.getLogger(__name__)

class YoutubeChunkingChain(Chain):
    """
    Chain that downloads YouTube audio, converts it to MP4, 
    and splits it into overlapping chunks

    Initialization Parameters:
    ----------
    chunk_length_ms : int
        The duration of each audio segment in milliseconds. Defaults to the
        value from `settings.CHUNK_LENGTH_MS`.
    overlap_ms : int
        The amount of overlap between consecutive segments in milliseconds.
        Defaults to the value from `settings.OVERLAP_MS`.
    base_dir : Path
        The base directory where audio files will be saved. Defaults to the
        value from `settings.DATA_DIR`.

    Attributes:
    ----------
    input_keys : List[str]
        A read-only property defining the expected input keys for the `_call`
        method. It expects the "_video_id" key.
    output_keys : List[str]
        A read-only property defining the keys of the output dictionary to be
        returned. It returns "_video_id" and "chunk_paths".
    
    """

    chunk_length_ms: int  = Field(default = settings.CHUNK_LENGTH_MS)
    overlap_ms:      int  = Field(default = settings.OVERLAP_MS)
    base_dir:        Path = Field(default = Path(settings.DATA_DIR))


    model_config = ConfigDict(extra="ignore") # ignore unexpected fields in input dictionary

    @property
    def input_keys(self) -> List[str]:
        return ["_video_id"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["_video_id", "chunk_paths"]
    
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    
    def _call(self, inputs: Dict) -> Dict:

        #create directories
        _video_id = inputs["_video_id"]
        full_dir    = self.base_dir / _video_id / "audios" / "full"
        chunks_dir  = self.base_dir / _video_id / "audios" / "chunks"
        full_dir.mkdir(parents=True, exist_ok=True)
        chunks_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directories to store audio files created successfully for video ID: {_video_id}")
        
        #download audio from video ID
        try:
            logger.debug(f"Downloading audio from video ID: {_video_id}...")
            loader = YoutubeAudioLoader(urls=[f"https://www.youtube.com/watch?v={_video_id}"], 
                                        save_dir=str(full_dir))
            blob   = next(loader.yield_blobs())
            src    = Path(blob.path)
            dst    = full_dir / f"{_video_id}{src.suffix}"
            src.rename(dst)
            logger.debug(f"Audios downloaded successfully for video ID: {_video_id}")
        
        except Exception as e:
            logger.exception(f"Error downloading audio for video ID {_video_id}: {e}")
            raise e
        
        # convert to mp4
        try: 
            logger.debug(f"Converting audio to MP4 format for video ID: {_video_id}...")
            audio = AudioSegment.from_file(dst, format=dst.suffix.lstrip("."))
            audio_path = full_dir / f"{_video_id}.mp4"
            audio.export(audio_path, format="mp4")
            dst.unlink()
            logger.debug(f"Audio converted successfully to MP4 format for video ID: {_video_id}")
        
        except Exception as e:
            logger.exception(f"Audio conversion/export failed for video ID {_video_id}: %s", e)
            raise e

        # chunking videos with overlaping
        step  = self.chunk_length_ms - self.overlap_ms
        total = len(audio)
        chunk_paths: List[str] = []

        try:
            logger.debug(f"Chunking audios into {self.chunk_length_ms} ms chunks with {self.overlap_ms} ms overlap for video ID: {_video_id}...")
            for idx, start_ms in enumerate(range(0, total, step)):
                end_ms  = min(start_ms + self.chunk_length_ms, total)
                chunk   = audio[start_ms:end_ms]
                fname   = f"{_video_id}_chunk_{idx}.mp4"
                outpath = chunks_dir / fname
                chunk.export(outpath, format="mp4")
                chunk_paths.append(str(outpath))
                logger.debug(f"Chunk {idx}: {start_ms}–{end_ms} ms → {fname}")
            logger.debug(f"Chunking completed successfully. {len(chunk_paths)} chunks created for video ID: {_video_id}.")
        
        except Exception as e:
            logger.exception(f"Unexpected error during chunking for video ID {_video_id}: %s", e)
            raise e
        
        return {
            "_video_id": _video_id,
            "chunk_paths": chunk_paths
            }
        