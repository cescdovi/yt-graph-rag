from pathlib import Path
from yt_neo4j_etl.src.chains.video_chunking import YoutubeChunkingChain

class FakeBlob:
    def __init__(self, path): self.path = path

class LoaderOK:
    def __init__(self, urls, save_dir):
        self.save_dir = save_dir
    def yield_blobs(self):
        p = Path(self.save_dir) / "dl.webm"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        yield FakeBlob(str(p))

class AudioStub:
    def __init__(self, total=10_000): self.total = total
    def __len__(self): return self.total
    def __getitem__(self, _): return self
    def export(self, out, format): Path(out).write_bytes(b"x")

class AudioSegmentStub:
    @classmethod
    def from_file(cls, filepath, format):
        return AudioStub(total=10_000)  # 10s

def test_happy_path_simple(monkeypatch, tmp_path):
    monkeypatch.setattr("yt_neo4j_etl.src.chains.video_chunking.YoutubeAudioLoader", LoaderOK)
    monkeypatch.setattr("yt_neo4j_etl.src.chains.video_chunking.AudioSegment", AudioSegmentStub)

    chain = YoutubeChunkingChain(
        chunk_length_ms=4000,  # 4s
        overlap_ms=1000,       # 1s -> step=3000
        base_dir=tmp_path,
    )

    out = chain._call({"_video_id": "vid"})
    assert out["_video_id"] == "vid"

    assert len(out["chunk_paths"]) == 4
    assert [Path(p).name for p in out["chunk_paths"]] == [
        "vid_chunk_0.mp4", "vid_chunk_1.mp4", "vid_chunk_2.mp4", "vid_chunk_3.mp4"
    ]

    # check if exists
    for p in out["chunk_paths"]:
        assert Path(p).exists()
