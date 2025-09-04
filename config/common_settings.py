from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, Field, model_validator
class Settings(BaseSettings):
    #api keys
    OPENAI_API_KEY: str
    YOUTUBE_API_KEY: str
    TAVILY_API_KEY: str
    
    #NEO4J
    NEO4J_URI_BOLT: str
    NEO4J_URI_HTTP: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str

    #youtube playlist id
    PLAYLIST_ID: str

    #directories
    DATA_DIR: str = "data"

    #chunking and overlap settings
    CHUNK_LENGTH_MS: int = Field(..., ge=1000, le=3600000)  
    OVERLAP_MS: int = Field(..., ge=0, le=600000)

    #models 
    TRANSCRIPTION_MODEL: str
    LLM_MODEL: str

    MAX_RETRIES:int = Field(..., ge=0)

    # load env variables from .env file
    model_config = SettingsConfigDict(env_file=".env")
    
   
    
    @field_validator(
            "OPENAI_API_KEY", 
            "YOUTUBE_API_KEY",
            "TAVILY_API_KEY", 
            "NEO4J_URI_BOLT",
            "NEO4J_URI_HTTP",
            "NEO4J_USER",
            "NEO4J_PASSWORD",
            "NEO4J_DATABASE",
            "PLAYLIST_ID", 
            "TRANSCRIPTION_MODEL", 
            "DATA_DIR")
    def validate_openai_api_key(cls, v, field):
        if not v.strip():
            raise ValueError(f"{field.name} cannot be empty")
        return v

    @model_validator(mode="after")
    def validate_chunking(self):
        #overlap must be less than chunk length
        if self.OVERLAP_MS >= self.CHUNK_LENGTH_MS:
            raise ValueError(
                "OVERLAP_MS must be strictly less than CHUNK_LENGTH_MS "
                f"(got {self.OVERLAP_MS} ≥ {self.CHUNK_LENGTH_MS})."
            )

        step = self.CHUNK_LENGTH_MS - self.OVERLAP_MS

        # recommended: avoid overlap > 70% of chunk length
        if self.OVERLAP_MS > int(self.CHUNK_LENGTH_MS * 0.7):
            raise ValueError(
                f"OVERLAP_MS is excessively large (> 80% of CHUNK_LENGTH_MS). "
                f"Got overlap={self.OVERLAP_MS} ms, chunk={self.CHUNK_LENGTH_MS} ms."
            )

        return self

settings = Settings()