import logging
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pathlib import Path
import json 

from config.common_settings import settings

# Configure logging
logger = logging.getLogger(__name__)


def get_youtube_client():
    """Creates a YouTube client using the provided API key."""
    try:
        return build(
            serviceName="youtube",
            version="v3",
            developerKey=settings.YOUTUBE_API_KEY
        )
    except Exception as e:
        logger.exception("Error creating YouTube client with the provided API")
        raise RuntimeError("Error creating YouTube client with the provided API") from e


def get_urls_from_playlist(
        playlist_id = settings.PLAYLIST_ID, 
        ):
    """Extracts video URLs from a YouTube playlist and saves metadata to JSON files."""
    
    logger.debug("Creating YouTube client...")
    youtube = get_youtube_client()
    logger.debug("Youtube client created successfully")

    videos = []
    next_page_token = None

    #extract general info about videos in the playlist
    while True:
        try:
            request = youtube.playlistItems().list(
                part       = "contentDetails",
                playlistId = playlist_id,
                maxResults = 50,            
                pageToken  = next_page_token
            )
            response = request.execute()

            for item in response.get('items', []):
                vid = item['contentDetails']['videoId']
                videos.append(vid)

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

        except HttpError as e:
            logger.error(f"HTTP error while fetching videos from playlist {playlist_id}: {e}")
            raise 
        except Exception as e:
            logger.exception(f"An unexpected error occurred while fetching videos from playlist {playlist_id}: {e}")
            raise

    # Save metadata for each video
    for video in videos:
        try:
            videos_dict = {}
            request = youtube.videos().list(
                part="snippet",
                id=video
                )
            response = request.execute()
            items = response.get("items", [])
            if not items:
                logger.warning(f"No details found for video ID {video}")
                continue
            
            _video_id = items[0]["id"]
            title = items[0]["snippet"]["title"]
            description = items[0]["snippet"]["description"]

            videos_dict[_video_id] = {
                "title": title,
                "description": description
            }

            output_dir = Path(settings.DATA_DIR) / _video_id
            output_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = output_dir / "metadata.json"

            with open((metadata_path), "w", encoding="utf-8") as f:
                json.dump(videos_dict, f, ensure_ascii=False, indent=2)

        except HttpError as e:
            logger.error(f"Error while fetching video details for video ID {video}: {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error for video ID {video}: {e}")
            raise

    return videos