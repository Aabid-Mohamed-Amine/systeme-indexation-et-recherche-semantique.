"""
Migration script — importe snrt_metadata_v2.json dans MongoDB.

Regroupe les segments par source_audio (= une vidéo),
crée un document par vidéo dans la collection `videos`.

Usage:
    python migrate_to_mongo.py
    python migrate_to_mongo.py --mongo mongodb://localhost:27017
"""

import argparse
import json
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

META_PATH = Path(__file__).parent / "data" / "snrt_metadata_v2.json"


def migrate(mongo_uri: str, db_name: str = "SNRT_ARCHIVE", dry_run: bool = False):
    if not META_PATH.exists():
        print(f"[ERREUR] Fichier introuvable : {META_PATH}")
        sys.exit(1)

    print(f"[INFO] Lecture de {META_PATH} …")
    with open(META_PATH, encoding="utf-8") as f:
        all_segments: list[dict] = json.load(f)

    print(f"[INFO] {len(all_segments)} segments trouves")

    # ── Regrouper par source_audio ────────────────────────────────
    by_video: dict[str, list[dict]] = defaultdict(list)
    for seg in all_segments:
        source = seg.get("source_audio", "unknown")
        by_video[source].append(seg)

    print(f"[INFO] {len(by_video)} video(s) detectee(s) :\n")
    for src, segs in by_video.items():
        lang = segs[0].get("lang", "?")
        src_safe = src.encode("ascii", errors="replace").decode("ascii")
        print(f"  - {src_safe}  ({len(segs)} segments, lang={lang})")

    if dry_run:
        print("\n[DRY-RUN] Aucune écriture effectuée.")
        return

    # ── Connexion MongoDB ─────────────────────────────────────────
    try:
        from pymongo import MongoClient
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=4000)
        client.admin.command("ping")
        print(f"\n[OK] Connecté à {mongo_uri}")
    except Exception as e:
        print(f"\n[ERREUR] Connexion MongoDB impossible : {e}")
        print("  → Assurez-vous que MongoDB tourne (docker compose up mongo -d)")
        sys.exit(1)

    col = client[db_name]["videos"]
    col.create_index("video_id", unique=True)
    col.create_index("filename")

    inserted, updated = 0, 0

    for filename, segments in by_video.items():
        lang = segments[0].get("lang", "")

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
        result = col.replace_one({"video_id": video_id}, doc, upsert=True)

        if result.upserted_id:
            inserted += 1
            print(f"  [INSERT] {filename}  → video_id={video_id}")
        else:
            updated += 1
            print(f"  [UPDATE] {filename}  → video_id={video_id}")

    print(f"\n[DONE] {inserted} insérée(s), {updated} mise(s) à jour")
    print(f"       Base : {db_name}  |  Collection : videos")
    print(f"       Total documents : {col.count_documents({})}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate JSON metadata → MongoDB")
    parser.add_argument("--mongo",   default="mongodb://localhost:27017", help="URI MongoDB")
    parser.add_argument("--db",      default="SNRT_ARCHIVE",              help="Nom de la base")
    parser.add_argument("--dry-run", action="store_true",                 help="Simuler sans écrire")
    args = parser.parse_args()

    migrate(args.mongo, args.db, args.dry_run)
