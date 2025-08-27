import logging

from config.common_settings import settings
from config.setup_logging import setup_logging

from yt_neo4j_etl.src.extract_urls_from_playlist import get_urls_from_playlist


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

if __name__ == "__main__":
    main()