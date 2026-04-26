#!/usr/bin/env python3
"""
Ingest all OCR text files into a SQLite database with FTS5 full-text search.

Schema:
  letters     — one row per OCR file, with parsed metadata
  letters_fts — FTS5 virtual table for full-text search over body + metadata
"""

import re
import sqlite3
from pathlib import Path

DATA_DIR = Path("data/Ackerman")
DB_PATH = Path("letters.db")

MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}


def parse_date(raw: str):
    """Return ISO date string (YYYY-MM-DD, YYYY-MM, or YYYY) or None."""
    raw = raw.strip()

    # YYYY-MM-DD
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        return raw

    # YYYY-MM
    m = re.fullmatch(r"(\d{4})-(\d{2})", raw)
    if m:
        return raw

    # YYYY
    m = re.fullmatch(r"\d{4}", raw)
    if m:
        return raw

    # "Nov 5 1938" / "May 26 1935"
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2})\s+(\d{4})", raw)
    if m:
        month = MONTH_MAP.get(m.group(1).lower()[:3])
        if month:
            return f"{m.group(3)}-{month}-{int(m.group(2)):02d}"

    # "Apr 1932" (month + year, no day)
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{4})", raw)
    if m:
        month = MONTH_MAP.get(m.group(1).lower()[:3])
        if month:
            return f"{m.group(2)}-{month}"

    return None


def parse_stem(stem: str) -> dict:
    """
    Parse an OCR filename stem (without _ocr.txt) into structured fields.

    Handles:
      Sender - Recipient - Date
      Sender - Recipient - Date - Page
      Sender - Recipient - Date_PageNum     (Syracuse underscore style)
      Sender - Recipient - Date (draft)
    """
    is_draft = bool(re.search(r"\(draft\)", stem, re.IGNORECASE))
    clean = re.sub(r"\s*\(draft\)", "", stem, flags=re.IGNORECASE).strip()

    parts = [p.strip() for p in clean.split(" - ")]

    if len(parts) < 3:
        return dict(sender=None, recipient=None, date_raw=None, date=None,
                    page=None, is_draft=is_draft)

    sender = parts[0]
    recipient = parts[1]
    date_field = parts[2]
    page = parts[3] if len(parts) >= 4 else None

    # Handle underscore-embedded page: "1947-01-18_02" or "Date_2"
    if page is None:
        m = re.search(r"_(\w+)$", date_field)
        if m and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_field):
            page = m.group(1)
            date_field = date_field[: m.start()]

    return dict(
        sender=sender,
        recipient=recipient,
        date_raw=date_field,
        date=parse_date(date_field),
        page=page,
        is_draft=is_draft,
    )


def create_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS letters (
            id          INTEGER PRIMARY KEY,
            filename    TEXT NOT NULL,
            filepath    TEXT NOT NULL UNIQUE,
            subfolder   TEXT,
            sender      TEXT,
            recipient   TEXT,
            date_raw    TEXT,
            date        TEXT,
            page        TEXT,
            is_draft    INTEGER DEFAULT 0,
            body        TEXT NOT NULL,
            ingested_at TEXT DEFAULT (datetime('now'))
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS letters_fts USING fts5(
            sender,
            recipient,
            date_raw,
            body,
            content=letters,
            content_rowid=id
        );

        CREATE TRIGGER IF NOT EXISTS letters_ai AFTER INSERT ON letters BEGIN
            INSERT INTO letters_fts(rowid, sender, recipient, date_raw, body)
            VALUES (new.id, new.sender, new.recipient, new.date_raw, new.body);
        END;
    """)
    conn.commit()
    return conn


def ingest(data_dir: Path = DATA_DIR, db_path: Path = DB_PATH) -> None:
    ocr_files = sorted(data_dir.rglob("*_ocr.txt"))
    print(f"Found {len(ocr_files)} OCR files in {data_dir}")

    conn = create_db(db_path)
    inserted = skipped = errors = 0

    for path in ocr_files:
        rel = path.relative_to(data_dir)
        subfolder = str(rel.parent) if rel.parent != Path(".") else None

        stem = path.name.removesuffix("_ocr.txt")
        fields = parse_stem(stem)

        try:
            body = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  ERROR reading {path.name}: {e}")
            errors += 1
            continue

        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO letters
                    (filename, filepath, subfolder, sender, recipient,
                     date_raw, date, page, is_draft, body)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    path.name,
                    str(path),
                    subfolder,
                    fields["sender"],
                    fields["recipient"],
                    fields["date_raw"],
                    fields["date"],
                    fields["page"],
                    1 if fields["is_draft"] else 0,
                    body,
                ),
            )
            if conn.execute("SELECT changes()").fetchone()[0]:
                inserted += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  ERROR inserting {path.name}: {e}")
            errors += 1

    conn.commit()
    conn.close()

    print(f"Done: {inserted} inserted, {skipped} already existed, {errors} errors")
    print(f"Database written to: {db_path}")


if __name__ == "__main__":
    ingest()
