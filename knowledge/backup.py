from __future__ import annotations

import zipfile
from datetime import datetime, timezone
from pathlib import Path

from config import DATA_DIR, CLOUD_SYNC_FOLDER, MAX_BACKUPS


def create_backup() -> dict:
    """
    Zip the data/ directory (SQLite db + ChromaDB store) into a timestamped
    archive inside CLOUD_SYNC_FOLDER. OneDrive's own desktop client handles
    the actual cloud upload -- this function only needs to create the zip
    locally inside the synced folder.

    Arcnames are stored relative to DATA_DIR.parent (the project root), so
    that knowledge/restore.py can extract them back into the correct place
    with zf.extractall(project_root).
    """
    data_dir = Path(DATA_DIR)
    project_root = data_dir.parent

    if not data_dir.exists():
        raise FileNotFoundError(f"No data directory found at {data_dir}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    zip_name = f"nova_backup_{stamp}.zip"
    zip_path = CLOUD_SYNC_FOLDER / zip_name

    file_count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in data_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(project_root)
                zf.write(file_path, arcname)
                file_count += 1

    _prune_old_backups()

    return {
        "backup_name": zip_name,
        "backup_path": str(zip_path),
        "files_backed_up": file_count,
        "size_mb": round(zip_path.stat().st_size / (1024 * 1024), 2),
    }


def list_backups() -> list[dict]:
    """Return all backups in CLOUD_SYNC_FOLDER, newest first."""
    backups = []
    for zip_path in CLOUD_SYNC_FOLDER.glob("nova_backup_*.zip"):
        backups.append({
            "name": zip_path.name,
            "path": str(zip_path),
            "size_mb": round(zip_path.stat().st_size / (1024 * 1024), 2),
            "created": datetime.fromtimestamp(
                zip_path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        })
    backups.sort(key=lambda b: b["created"], reverse=True)
    return backups


def _prune_old_backups() -> None:
    """Keep only the MAX_BACKUPS most recent backups, delete the rest."""
    backups = list_backups()
    for old in backups[MAX_BACKUPS:]:
        Path(old["path"]).unlink()