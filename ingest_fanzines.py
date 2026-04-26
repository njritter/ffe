#!/usr/bin/env python3
"""
Ingest fanzine OCR text files into a SQLite database with FTS5 full-text search.

Metadata is derived from the directory path:
  Fanzines/[FanzineName]/[IssueFolder]/[PageFile]

Schema:
  pages     — one row per OCR file
  pages_fts — FTS5 virtual table for full-text search
"""

import re
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
import os

load_dotenv()

DB_PATH = Path("fanzines.db")

MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}

SEASON_MAP = {
    "spring": "03", "summer": "06", "fall": "09", "winter": "12",
}

ISSUE_CODE_RE = re.compile(r'\b(v(\d+)n(\d+)|n(\d+)|v(\d+))\b', re.IGNORECASE)


def parse_date(raw: str):
    """Return ISO date string (YYYY-MM-DD, YYYY-MM, or YYYY) or None."""
    raw = raw.strip()

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw
    if re.fullmatch(r"\d{4}-\d{2}", raw):
        return raw
    if re.fullmatch(r"\d{4}", raw):
        return raw

    # "Dec 1 1943" / "May 26 1935"
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2})\s+(\d{4})", raw)
    if m:
        month = MONTH_MAP.get(m.group(1).lower()[:3])
        if month:
            return f"{m.group(3)}-{month}-{int(m.group(2)):02d}"

    # "May 1939" / "Apr 1932"
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{4})", raw)
    if m:
        month = MONTH_MAP.get(m.group(1).lower()[:3])
        if month:
            return f"{m.group(2)}-{month}"

    return None


def extract_date_from_text(text: str):
    """Find the first date-like substring in text. Returns (raw, normalized)."""
    # Season + year: "Fall 1942"
    m = re.search(r'\b(Spring|Summer|Fall|Winter)\s+(\d{4})\b', text, re.IGNORECASE)
    if m:
        season_month = SEASON_MAP[m.group(1).lower()]
        return m.group(0), f"{m.group(2)}-{season_month}"

    # Month day year: "Dec 1 1943"
    m = re.search(r'\b([A-Za-z]{3,9})\s+(\d{1,2})\s+(\d{4})\b', text)
    if m:
        key = m.group(1).lower()[:3]
        if key in MONTH_MAP:
            return m.group(0), f"{m.group(3)}-{MONTH_MAP[key]}-{int(m.group(2)):02d}"

    # Month year: "May 1939"
    m = re.search(r'\b([A-Za-z]{3,9})\s+(\d{4})\b', text)
    if m:
        key = m.group(1).lower()[:3]
        if key in MONTH_MAP:
            return m.group(0), f"{m.group(2)}-{MONTH_MAP[key]}"

    # Bare year
    m = re.search(r'\b(1[89]\d{2}|20[012]\d)\b', text)
    if m:
        return m.group(0), m.group(0)

    return None, None


def parse_issue_folder(folder_name: str) -> dict:
    """Extract issue_code, volume, issue_number, date_raw, date from an issue folder name."""
    volume = issue_number = issue_code = None

    m = ISSUE_CODE_RE.search(folder_name)
    if m:
        issue_code = m.group(0).lower()
        if m.group(2) is not None:   # v#n# form
            volume = int(m.group(2))
            issue_number = int(m.group(3))
        elif m.group(4) is not None:  # n# form
            issue_number = int(m.group(4))
        elif m.group(5) is not None:  # v# form
            volume = int(m.group(5))

    date_raw, date = extract_date_from_text(folder_name)
    return {
        "issue_code": issue_code,
        "volume": volume,
        "issue_number": issue_number,
        "date_raw": date_raw,
        "date": date,
    }


def read_provenance(issue_dir: Path) -> str | None:
    """Return text of first non-OCR .txt file in the issue directory, or None."""
    for f in issue_dir.iterdir():
        if f.is_file() and f.suffix == ".txt" and not f.name.endswith("_ocr.txt"):
            try:
                return f.read_text(encoding="utf-8", errors="replace").strip() or None
            except Exception:
                pass
    return None


def extract_page(stem: str) -> str:
    """Extract page identifier from a filename stem (without _ocr.txt)."""
    parts = [p.strip() for p in stem.split(" - ")]
    last = parts[-1] if parts else stem

    # Strip leading "p" prefix: "p01" → "01"
    m = re.fullmatch(r"[pP](\d+.*)", last)
    if m:
        return m.group(1)

    # Bare number: "01", "0001"
    if re.fullmatch(r"\d+", last):
        return last

    return last


def create_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS pages (
            id           INTEGER PRIMARY KEY,
            filename     TEXT NOT NULL,
            filepath     TEXT NOT NULL UNIQUE,
            fanzine      TEXT,
            issue_folder TEXT,
            issue_code   TEXT,
            volume       INTEGER,
            issue_number INTEGER,
            date_raw     TEXT,
            date         TEXT,
            page         TEXT,
            subfolder    TEXT,
            provenance   TEXT,
            body         TEXT NOT NULL,
            ingested_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
            fanzine,
            issue_folder,
            date_raw,
            provenance,
            body,
            content=pages,
            content_rowid=id
        );

        CREATE TRIGGER IF NOT EXISTS pages_ai AFTER INSERT ON pages BEGIN
            INSERT INTO pages_fts(rowid, fanzine, issue_folder, date_raw, provenance, body)
            VALUES (new.id, new.fanzine, new.issue_folder, new.date_raw, new.provenance, new.body);
        END;
    """)
    conn.commit()
    return conn


def ingest(data_dir: Path, db_path: Path = DB_PATH) -> None:
    if not data_dir.exists():
        print(f"Error: directory not found: {data_dir}")
        sys.exit(1)

    ocr_files = sorted(data_dir.rglob("*_ocr.txt"))
    total = len(ocr_files)
    print(f"Found {total} OCR files in {data_dir}")

    conn = create_db(db_path)
    inserted = skipped = errors = 0

    # Cache provenance per issue directory to avoid re-reading
    provenance_cache: dict[Path, str | None] = {}

    with tqdm(total=total, unit="file", dynamic_ncols=True) as pbar:
        for i, path in enumerate(ocr_files, 1):
            rel = path.relative_to(data_dir)
            parts = rel.parts  # (fanzine, [issue_folder], [subfolder...], filename)

            fanzine = parts[0] if len(parts) >= 1 else None

            if len(parts) >= 3:
                issue_folder = parts[1]
                subfolder_parts = parts[2:-1]
            else:
                issue_folder = None
                subfolder_parts = []

            subfolder = "/".join(subfolder_parts) if subfolder_parts else None

            if issue_folder:
                parsed = parse_issue_folder(issue_folder)
                issue_code   = parsed["issue_code"]
                volume       = parsed["volume"]
                issue_number = parsed["issue_number"]
                date_raw     = parsed["date_raw"]
                date         = parsed["date"]

                issue_dir = data_dir / fanzine / issue_folder
                if issue_dir not in provenance_cache:
                    provenance_cache[issue_dir] = read_provenance(issue_dir)
                provenance = provenance_cache[issue_dir]
            else:
                issue_code = volume = issue_number = date_raw = date = provenance = None

            stem = path.name.removesuffix("_ocr.txt")
            page = extract_page(stem)

            try:
                body = path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                tqdm.write(f"  ERROR reading {path.name}: {e}")
                errors += 1
                pbar.update(1)
                continue

            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO pages
                        (filename, filepath, fanzine, issue_folder, issue_code,
                         volume, issue_number, date_raw, date, page,
                         subfolder, provenance, body)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        path.name,
                        str(path),
                        fanzine,
                        issue_folder,
                        issue_code,
                        volume,
                        issue_number,
                        date_raw,
                        date,
                        page,
                        subfolder,
                        provenance,
                        body,
                    ),
                )
                if conn.execute("SELECT changes()").fetchone()[0]:
                    inserted += 1
                else:
                    skipped += 1
            except Exception as e:
                tqdm.write(f"  ERROR inserting {path.name}: {e}")
                errors += 1

            if i % 1000 == 0:
                conn.commit()

            pbar.update(1)
            pbar.set_postfix(inserted=inserted, skipped=skipped, errors=errors)

    conn.commit()
    conn.close()

    print(f"\nDone: {inserted} inserted, {skipped} already existed, {errors} errors")
    print(f"Database written to: {db_path.resolve()}")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.getenv("ROOT_DIRECTORY", ""))
    if not root or str(root) == ".":
        print("Usage: python ingest_fanzines.py <path-to-fanzines-dir>")
        print("  or set ROOT_DIRECTORY in .env")
        sys.exit(1)
    ingest(root)
