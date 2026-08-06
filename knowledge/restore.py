from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

from config import DATA_DIR


def _resolve_backup_path(name_or_path: str) -> Path:
    """Accept either a bare filename ('nova_backup_20260806_213000.zip')
    or a full/relative path, and return a valid Path to the zip."""
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate

    candidate = CLOUD_SYNC_FOLDER / name_or_path
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Could not find backup '{name_or_path}' as a path or inside {CLOUD_SYNC_FOLDER}"
    )


def verify_backup(backup_path: Path) -> bool:
    """Check the zip isn't corrupted before touching anything on disk."""
    if not zipfile.is_zipfile(backup_path):
        print(f"'{backup_path}' is not a valid zip file.")
        return False

    with zipfile.ZipFile(backup_path, "r") as zf:
        bad_file = zf.testzip()
        if bad_file is not None:
            print(f"Corrupt file inside archive: {bad_file}")
            return False

    return True


def restore_backup(name_or_path: str, force: bool = False) -> dict:
    """
    Restore a backup zip into place.

    Safety behavior:
    - Verifies the archive integrity first.
    - If the current data/ dir exists and isn't empty, it is renamed to
      data_pre_restore_<timestamp>/ instead of being deleted, so a bad
      restore can't destroy anything irreversibly.
    - Extracts into DATA_DIR.parent, matching how backup.py wrote arcnames
      (relative to DATA_DIR.parent), so paths land back at data/... .
    """
    backup_path = _resolve_backup_path(name_or_path)
    project_root = Path(DATA_DIR).parent
    data_dir = Path(DATA_DIR)

    print(f"Verifying archive: {backup_path}")
    if not verify_backup(backup_path):
        raise ValueError("Backup failed verification. Restore aborted.")

    if data_dir.exists() and any(data_dir.iterdir()) and not force:
        from datetime import datetime, timezone

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safety_dir = project_root / f"data_pre_restore_{stamp}"
        print(f"Existing data/ found. Moving it to {safety_dir} before restoring.")
        shutil.move(str(data_dir), str(safety_dir))

    project_root.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {backup_path.name} into {project_root} ...")
    with zipfile.ZipFile(backup_path, "r") as zf:
        zf.extractall(project_root)

    print("Restore complete.")
    return {
        "restored_from": str(backup_path),
        "restored_to": str(data_dir),
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Restore a NOVA backup archive.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List available backups.")

    verify_p = sub.add_parser("verify", help="Verify a backup archive's integrity.")
    verify_p.add_argument("backup", help="Backup filename or path.")

    restore_p = sub.add_parser("restore", help="Restore from a backup archive.")
    restore_p.add_argument("backup", help="Backup filename or path.")
    restore_p.add_argument(
        "--force",
        action="store_true",
        help="Skip the safety copy of the current data/ directory (not recommended).",
    )

    args = parser.parse_args()

    if args.command == "list":
        backups = list_backups()
        if not backups:
            print(f"No backups found in {CLOUD_SYNC_FOLDER}")
            return
        print(f"Backups in {CLOUD_SYNC_FOLDER}:")
        for b in backups:
            print(f"  {b['name']}  ({b['size_mb']} MB)")

    elif args.command == "verify":
        path = _resolve_backup_path(args.backup)
        ok = verify_backup(path)
        print("OK" if ok else "FAILED")
        sys.exit(0 if ok else 1)

    elif args.command == "restore":
        try:
            result = restore_backup(args.backup, force=args.force)
            print(result)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    _cli()
