import logging
from pathlib import Path

from config.common_settings import settings
from config.setup_logging import setup_logging

from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser

from yt_neo4j_etl.src.extract_urls_from_playlist import get_urls_from_playlist
from yt_neo4j_etl.src.chains.video_chunking import YoutubeChunkingChain
from yt_neo4j_etl.src.chains.transcription import WhisperTranscriptionChain

from yt_neo4j_etl.src.prompts.prompt_transcription import chat_prompt_transcription

def main():
    setup_logging() 
    logger = logging.getLogger(__name__)

    logger.info("Starting ETL load to Neo4j")
    logger.info("-" * 60)

    logger.info(" Starting URL extraction from YouTube playlist")
    urls = get_urls_from_playlist(settings.PLAYLIST_ID)
    logger.info(" URL extraction completed")
    logger.info(f" Extracted {len(urls)} URLs")

    logger.info("-" * 60)
    logger.info(f"URLS: {urls}")

    logger.info("Starting video chunking")
    
    urls = urls[:1]
    chunk_chain = YoutubeChunkingChain(
        chunk_length_ms = settings.CHUNK_LENGTH_MS,
        overlap_ms = settings.OVERLAP_MS,
        base_dir = Path(settings.DATA_DIR),
    )
    results_chunk_chain = [] # list of dicts
    for i in range(len(urls)):
        logger.info(f"Video URL: {urls[i]}")
        intermediate_chunking = chunk_chain.invoke({"_video_id": urls[i]})
        results_chunk_chain.append(intermediate_chunking)
    logger.info(f"Video chunking completed for {len(urls)} videos")
    
    logger.info("-" * 60)


    logger.info("Starting Whisper transcription...")
    parser = OpenAIWhisperParser(
        api_key = settings.OPENAI_API_KEY,
        model   = settings.TRANSCRIPTION_MODEL,
        prompt  = chat_prompt_transcription,
    )
    
    transcription_chain = WhisperTranscriptionChain(parser=parser)

    results_transcription_chain = []

    for i in range(len(results_chunk_chain)):
        _video_id   = results_chunk_chain[i]["_video_id"]
        chunk_paths = results_chunk_chain[i]["chunk_paths"]

        logger.info("Transcribing video_id=%s with %d chunks...", _video_id, len(chunk_paths))
        intermediate_transcription = transcription_chain.invoke({
            "_video_id": _video_id,
            "chunk_paths": [chunk_path for chunk_path in chunk_paths]
        })
        results_transcription_chain.append(intermediate_transcription)
    logger.info("All transcriptions completed.")


if __name__ == "__main__":
    main()