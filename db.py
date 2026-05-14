"""
MongoDB layer — stores one document per video with its segments embedded.

Collection `videos`:
{
  "_id": ObjectId,
  "video_id": str,          # UUID stable, used as foreign key
  "filename": str,          # original file name (e.g. "reportage.mp4")
  "lang": str,              # "fr" | "es" | "ar"
  "indexed_at": datetime,
  "n_segments": int,
  "segments": [             # full transcript, one entry per window
    {
      "segment_id": int,
      "text": str,
      "text_brut": str,
      "timecode": str,
      "timecode_end": str,
      "start_sec": float,
      "end_sec": float,
      "lang": str,
      "window_size": int
    }, ...
  ]
}
"""

import uuid
import logging
from datetime import datetime, timezone

from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure

from config import MONGO_URI, MONGO_DB

log = logging.getLogger(__name__)

_client: MongoClient | None = None


def _db():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    return _client[MONGO_DB]


def is_available() -> bool:
    try:
        _db().command("ping")
        return True
    except Exception:
        return False


def _ensure_indexes():
    col = _db()["videos"]
    col.create_index("video_id", unique=True)
    col.create_index("filename")
    col.create_index([("segments.source_audio", ASCENDING)])


# ── Write ─────────────────────────────────────────────────────────

def save_video(filename: str, lang: str, segments: list) -> str:
    """
    Insert or replace a video document.
    Returns the video_id (UUID string).
    """
    _ensure_indexes()
    col = _db()["videos"]

    # Reuse existing video_id if the file was already indexed before
    existing = col.find_one({"filename": filename}, {"video_id": 1})
    video_id = existing["video_id"] if existing else str(uuid.uuid4())

    doc = {
        "video_id":   video_id,
        "filename":   filename,
        "lang":       lang,
        "indexed_at": datetime.now(timezone.utc),
        "n_segments": len(segments),
        "segments":   segments,
    }
    col.replace_one({"video_id": video_id}, doc, upsert=True)
    log.info("MongoDB: saved video %s (%d segments)", filename, len(segments))
    return video_id


# ── Read ──────────────────────────────────────────────────────────

def get_all_segments() -> list[dict]:
    """Return a flat list of all segments from every video (mirrors the JSON metadata format)."""
    col = _db()["videos"]
    result = []
    for video in col.find({}, {"segments": 1, "video_id": 1, "_id": 0}):
        for seg in video.get("segments", []):
            result.append({**seg, "video_id": video["video_id"]})
    return result


def get_video(video_id: str) -> dict | None:
    return _db()["videos"].find_one({"video_id": video_id}, {"_id": 0})


def list_videos() -> list[dict]:
    """Return lightweight video list (no segments)."""
    col = _db()["videos"]
    return list(col.find({}, {"_id": 0, "segments": 0}))


def delete_video(video_id: str) -> bool:
    result = _db()["videos"].delete_one({"video_id": video_id})
    return result.deleted_count > 0
