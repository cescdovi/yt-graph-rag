import logging
from typing import Dict, List
from pathlib import Path
from pydantic import PrivateAttr, ConfigDict
from langchain.chains.base import Chain
from langchain_core.runnables import RunnableSequence

# Set up a logger for the chain
logger = logging.getLogger(__name__)

class TranslationChain(Chain):
    """
    Chain that reasons if the text is in Valencian and translates it to Spanish.
    If it is in Spanish, it returns the text as is.

    Purpose
    -------
    - Takes a `video_id` and a text from the previous chain.
    - Uses a `detect_chain` to determine the language of the text.
    - If the language is Valencian, it uses a `translate_chain` to convert it to Spanish.
    - If the language is Spanish, it passes the text through unchanged.
    - Saves the final Spanish text to a file.
    - Returns the final Spanish text.

    Attributes
    ----------
    input_keys : List[str]
        Expected input keys: `["_video_id", "correference_resolution_text"]`.
    output_keys : List[str]
        Returned output keys: `["_video_id", "spanish_text"]`.
    """

    _detect_chain: RunnableSequence = PrivateAttr()
    _translate_chain: RunnableSequence = PrivateAttr()
    model_config = ConfigDict(extra="ignore") # ignore unexpected fields in input dictionary
    

    @property
    def input_keys(self) -> List[str]:
        return ["_video_id", "correference_resolution_text"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["_video_id", "spanish_text"]
    
    def __init__(self,
                 detect_chain: RunnableSequence,
                 translate_chain: RunnableSequence,
                 **kwargs):
        super().__init__(**kwargs)
        self._detect_chain = detect_chain
        self._translate_chain = translate_chain
        logger.info("TranslationChain initialized.")
    
    def _call(self, inputs: Dict) -> Dict:
        _video_id = inputs["_video_id"]
        original_text = inputs.get("correference_resolution_text")

        if not original_text:
            logger.warning("No text provided for translation for video_id=%s. Returning empty result.", _video_id)
            return {
                "_video_id": _video_id,
                "spanish_text": ""
            }

        # NOTE: `settings` is not a standard Python import. Assuming it's defined elsewhere.
        try:
            from config.common_settings import settings
            output_dir = Path(settings.DATA_DIR) / _video_id / "texts" / "spanish_text"
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"spanish_text_{_video_id}.txt"

        except ImportError:
            logger.error("Could not import `settings`. File saving will be skipped.")
            output_dir = None
            filename = None

        logger.info("Starting language detection for video_id=%s.", _video_id)
        # Get the language of the input text
        try:
            result_language = self._detect_chain.invoke({"text_to_dect": original_text})
            language = result_language.content.lower().strip()
            logger.info("Language detected: '%s'", language)
        except Exception as e:
            logger.error(
                "Error during language detection for video_id=%s: %s",
                _video_id,
                e,
                exc_info=True
            )
            raise

        if language == "valenciano":
            logger.info("Text is in Valencian. Starting translation to Spanish.")
            try:
                result_translation = self._translate_chain.invoke({"text_to_translate": original_text})
                translation = result_translation.content
                logger.debug("Translation completed. Original length: %d, Translated length: %d",
                             len(original_text), len(translation))
            except Exception as e:
                logger.error(
                    "Error during translation for video_id=%s: %s",
                    _video_id,
                    e,
                    exc_info=True
                )
                raise
            
            final_text = translation
        
        elif language == "castellano":
            logger.info("Text is in Spanish. No translation needed.")
            final_text = original_text
        
        else:
            logger.error("Unsupported language detected for video_id=%s: '%s'", _video_id, language)
            raise ValueError(f"Unsupported language: {language}")

        # Save the final text to a file
        if output_dir and filename:
            try:
                file_path = output_dir / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(final_text)
                logger.info("Translated/original text saved to file: %s", file_path)
            except Exception as e:
                logger.error(
                    "Error saving final text for video_id=%s: %s",
                    _video_id,
                    e,
                    exc_info=True
                )

        logger.info("TranslationChain finished for video_id=%s.", _video_id)
        return {
            "_video_id": _video_id,
            "spanish_text": final_text
        }
