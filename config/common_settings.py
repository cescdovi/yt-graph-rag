from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
class Settings(BaseSettings):
    #api keys
    OPENAI_API_KEY: str
    YOUTUBE_API_KEY: str

    #youtube playlist id
    PLAYLIST_ID: str

    #directories
    DATA_DIR: str = "data"

    # load env variables from .env file
    model_config = SettingsConfigDict(env_file=".env")
   
    
    @field_validator("OPENAI_API_KEY", "YOUTUBE_API_KEY", "PLAYLIST_ID", "DATA_DIR")
    def validate_openai_api_key(cls, v, field):
        if not v.strip():
            raise ValueError(f"{field.name} cannot be empty")
        return v

settings = Settings()