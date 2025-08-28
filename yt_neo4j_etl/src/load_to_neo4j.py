import logging
from pathlib import Path

from config.common_settings import settings
from config.setup_logging import setup_logging

from yt_neo4j_etl.src.extract_urls_from_playlist import get_urls_from_playlist
from yt_neo4j_etl.src.chains.video_chunking import YoutubeChunkingChain

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
    urls = urls[:2]
    chunk_chain = YoutubeChunkingChain(
        chunk_length_ms = settings.CHUNK_LENGTH_MS,
        overlap_ms = settings.OVERLAP_MS,
        base_dir = Path(settings.DATA_DIR),
    )
    results_chunk_chain = [] # list of dicts
    for i in range(len(urls)):
        logger.info(f"Video URL: {urls[i]}")
        intermediate_result = chunk_chain.invoke({"_video_id": urls[i]})
        results_chunk_chain.append(intermediate_result)
    logger.info(f"Video chunking completed for {len(urls)} videos")
    logger.info("-" * 60)



if __name__ == "__main__":
    main()