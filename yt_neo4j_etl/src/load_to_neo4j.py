from pathlib import Path
import logging
from config.common_settings import settings
from config.setup_logging import setup_logging

from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from yt_neo4j_etl.src.extract_urls_from_playlist import get_urls_from_playlist
from yt_neo4j_etl.src.chains.video_chunking import YoutubeChunkingChain
from yt_neo4j_etl.src.chains.transcription import WhisperTranscriptionChain
from yt_neo4j_etl.src.chains.unifiy_transcriptions import UnifyTranscriptsChain
from yt_neo4j_etl.src.chains.ortography_correction import OrtographyCorrectionChain
from yt_neo4j_etl.src.chains.correference_resolution import CorreferenceResolutionChain
from yt_neo4j_etl.src.chains.translation import TranslationChain

from yt_neo4j_etl.src.prompts.prompt_transcription import chat_prompt_transcription
from yt_neo4j_etl.src.prompts.prompt_unify_transcriptions import chat_prompt_unifier
from yt_neo4j_etl.src.prompts.prompt_ortography_correction import chat_prompt_corrector
from yt_neo4j_etl.src.prompts.prompt_correference_resolution import chat_prompt_correference_resolution
from yt_neo4j_etl.src.prompts.prompt_translation import chat_prompt_detect_language, chat_prompt_translation

def main():
    setup_logging()
    log = logging.getLogger(__name__)
    log.info("NEO4J ETL: Loading data from YouTube playlists to Neo4j...")

    # -- LLM y helpers
    llm = ChatOpenAI(model=settings.LLM_MODEL, api_key=settings.OPENAI_API_KEY, max_retries=settings.MAX_RETRIES)
    to_str = StrOutputParser()
    whisper = OpenAIWhisperParser(api_key=settings.OPENAI_API_KEY, model=settings.TRANSCRIPTION_MODEL, prompt=chat_prompt_transcription)

    # -- Chains atómicas
    chunk_chain   = YoutubeChunkingChain(chunk_length_ms=settings.CHUNK_LENGTH_MS, overlap_ms=settings.OVERLAP_MS, base_dir=Path(settings.DATA_DIR))
    transcription_chain   = WhisperTranscriptionChain(parser=whisper)
    unify_chain     = UnifyTranscriptsChain(unifier_chain=(chat_prompt_unifier | llm))
    correction_chain   = OrtographyCorrectionChain(corrective_chain=(chat_prompt_corrector | llm))
    corref_chain     = CorreferenceResolutionChain(correference_resolution_chain=(chat_prompt_correference_resolution | llm))
    translation_chain = TranslationChain(detect_chain=(chat_prompt_detect_language | llm),
                                 translate_chain=(chat_prompt_translation | llm))

    urls = get_urls_from_playlist(settings.PLAYLIST_ID)
    urls = urls[:1]  # para pruebas rápidas

    #chunking
    inputs_chunk_chain = [
        {
            "_video_id": vid
        } 
        for vid in urls
    ]
    results_chunk_chain = chunk_chain.batch(inputs_chunk_chain)

    #transcription
    inputs_transcription_chain = [
        {
            "_video_id": item["_video_id"],
            "chunk_paths": [chunk_path for chunk_path in item["chunk_paths"]]
        } 
        for item in results_chunk_chain
    ]
    results_transcription_chain = transcription_chain.batch(inputs_transcription_chain)

    #unify
    inputs_unify_chain = [
        {
            "_video_id": item["_video_id"],
            "transcripts": [transcript for transcript in item["transcripts"]]
        } 
        for item in results_transcription_chain
    ]
    results_unify_chain = unify_chain.batch(inputs_unify_chain)

    #correction
    inputs_correction_chain = [
        {
            "_video_id": item["_video_id"],
            "unified_transcript": item["unified_transcript"]
        } 
        for item in results_unify_chain
    ]
    results_correction_chain = correction_chain.batch(inputs_correction_chain)

    #Correference
    inputs_corref_chain = [
        {
            "_video_id": item["_video_id"],
            "corrected_text": item["corrected_text"]
        } 
        for item in results_correction_chain
    ]
    results_corref_chain = corref_chain.batch(inputs_corref_chain)

    #translation
    inputs_translation_chain = [
        {
            "_video_id": item["_video_id"],
            "correference_resolution_text": item["correference_resolution_text"]
        } 
        for item in results_corref_chain
    ]
    results_translation_chain = translation_chain.batch(inputs_translation_chain)

if __name__ == "__main__":
    main()
