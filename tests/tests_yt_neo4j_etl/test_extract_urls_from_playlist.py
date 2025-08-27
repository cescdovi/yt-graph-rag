import json
import pytest
from unittest.mock import MagicMock
from googleapiclient.errors import HttpError

# SUT
from yt_neo4j_etl.src import extract_urls_from_playlist as m


@pytest.fixture
def patch_settings(tmp_path, monkeypatch):
    """Apunta los ajustes del módulo a un tmp dir y valores fake."""
    monkeypatch.setattr(m.settings, "YOUTUBE_API_KEY", "test_key", raising=True)
    monkeypatch.setattr(m.settings, "PLAYLIST_ID", "test_playlist_id", raising=True)
    monkeypatch.setattr(m.settings, "DATA_DIR", str(tmp_path), raising=True)
    return tmp_path


def test_get_youtube_client_success(patch_settings, monkeypatch):
    fake_client = MagicMock(name="youtube")
    monkeypatch.setattr(m, "build", lambda **kw: fake_client, raising=True)

    client = m.get_youtube_client()
    assert client is fake_client


def test_get_youtube_client_failure_wraps_RuntimeError(patch_settings, monkeypatch):
    def boom(**kw):
        raise Exception("boom")
    monkeypatch.setattr(m, "build", boom, raising=True)

    with pytest.raises(RuntimeError):
        m.get_youtube_client()


def test_get_urls_from_playlist_success(patch_settings, monkeypatch):
    """Una sola página de playlist, un vídeo, y metadata OK."""
    yt = MagicMock(name="youtube")

    # Cadena playlistItems().list().execute()
    yt.playlistItems.return_value.list.return_value.execute.side_effect = [
        {
            "items": [{"contentDetails": {"videoId": "video1"}}],
            "nextPageToken": None
        }
    ]

    # Cadena videos().list().execute()
    yt.videos.return_value.list.return_value.execute.side_effect = [
        {
            "items": [{
                "id": "video1",
                "snippet": {"title": "Test Video", "description": "Test Desc"}
            }]
        }
    ]

    monkeypatch.setattr(m, "build", lambda **kw: yt, raising=True)

    result = m.get_urls_from_playlist()
    assert result == ["video1"]

    meta = (patch_settings / "video1" / "metadata.json")
    assert meta.exists()
    data = json.loads(meta.read_text(encoding="utf-8"))
    assert data["video1"]["title"] == "Test Video"
    assert data["video1"]["description"] == "Test Desc"


def test_get_urls_from_playlist_video_detail_http_error_continue(
    patch_settings, monkeypatch
):
    """Si falla el detalle de un vídeo con HttpError, debe continuar con los demás."""
    yt = MagicMock(name="youtube")

    yt.playlistItems.return_value.list.return_value.execute.side_effect = [
        {
            "items": [
                {"contentDetails": {"videoId": "a"}},
                {"contentDetails": {"videoId": "b"}},
            ],
            "nextPageToken": None
        }
    ]

    class _Resp:  # respuesta mínima para HttpError
        status = 500
        reason = "ERR"

    def raise_http_error(*args, **kwargs):
        raise HttpError(_Resp(), b"err")

    vids_exec = yt.videos.return_value.list.return_value.execute
    vids_exec.side_effect = [
        {"items": [{"id": "a", "snippet": {"title": "ta", "description": "da"}}]},
        HttpError(_Resp(), b"err"),
    ]

    monkeypatch.setattr(m, "build", lambda **kw: yt, raising=True)

    res = m.get_urls_from_playlist()
    assert res == ["a", "b"]
    assert (patch_settings / "a" / "metadata.json").exists()
    assert not (patch_settings / "b" / "metadata.json").exists()


def test_get_urls_from_playlist_playlist_http_error_raises(patch_settings, monkeypatch):
    """Si falla la llamada a playlistItems con HttpError, debe relanzarse."""
    yt = MagicMock(name="youtube")

    class _Resp:
        status = 403
        reason = "FORBIDDEN"

    def raise_http_error(*args, **kwargs):
        raise HttpError(_Resp(), b"forbidden")

    yt.playlistItems.return_value.list.return_value.execute.side_effect = [HttpError(_Resp(), b"forbidden")]
    monkeypatch.setattr(m, "build", lambda **kw: yt, raising=True)

    with pytest.raises(HttpError):
        m.get_urls_from_playlist()


def test_video_detail_items_vacio_warn(patch_settings, monkeypatch, caplog):
    """Cuando videos().list devuelve items vacío, loguea warning."""
    yt = MagicMock(name="youtube")
    yt.playlistItems.return_value.list.return_value.execute.side_effect = [
        {"items": [{"contentDetails": {"videoId": "x1"}}], "nextPageToken": None}
    ]
    yt.videos.return_value.list.return_value.execute.side_effect = [
        {"items": []}
    ]

    monkeypatch.setattr(m, "build", lambda **kw: yt, raising=True)
    caplog.set_level("WARNING")

    m.get_urls_from_playlist()

    assert any("No details found for video ID x1" in r.message for r in caplog.records)
