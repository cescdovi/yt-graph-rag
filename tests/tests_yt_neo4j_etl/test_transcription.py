import io
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from yt_neo4j_etl.src.chains.transcription import WhisperTranscriptionChain

@pytest.fixture
def tmp_chunks(tmp_path: Path):
    #create 2 chunks
    p1 = tmp_path / "chunk-001.wav"
    p2 = tmp_path / "chunk-002.wav"
    p1.write_bytes(b"fake-bytes-1")
    p2.write_bytes(b"fake-bytes-2")
    return [str(p1), str(p2)]

@pytest.fixture
def parser_mock():
    return MagicMock()

def make_docs(texts):
    return [Document(page_content=t) for t in texts]

def test_input_output_keys():
    chain = WhisperTranscriptionChain(parser=MagicMock())
    assert chain.input_keys == ["_video_id", "chunk_paths"]
    assert chain.output_keys == ["_video_id", "transcripts"]

def test_happy_path_multiple_chunks(tmp_chunks, parser_mock):
    parser_mock.parse.side_effect = [
        make_docs(["hola", "mundo"]),
        make_docs(["foo", "bar"]),
    ]
    chain = WhisperTranscriptionChain(parser=parser_mock)

    out = chain.invoke({
        "_video_id": "vid123",
        "chunk_paths": tmp_chunks,
    })

    assert out["_video_id"] == "vid123"
    # "hola mundo" and "foo bar" in the same order as the chunks
    assert out["transcripts"] == ["hola mundo", "foo bar"]

    assert parser_mock.parse.call_count == 2

    blob_arg = parser_mock.parse.call_args_list[0].args[0]
    assert getattr(blob_arg, "path", "").endswith("chunk-001.wav")
    assert blob_arg.metadata.get("chunk_filename") == "chunk-001.wav"
    assert blob_arg.data == Path(tmp_chunks[0]).read_bytes()