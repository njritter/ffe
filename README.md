# FFE Letters — OCR, Ingest & Search

A two-phase pipeline for digitising, ingesting, and searching a collection of historical science fiction fan letters (primarily Forrest J Ackerman correspondence, 1930s–1950s).

## Overview

```
Image scans (.jpg / .tif)
        │
        ▼
extract_text_parallel.py   ← Gemini Vision OCR → *_ocr.txt files
        │
        ▼
ingest.py                  ← Parse filenames + load text → letters.db (SQLite + FTS5)
        │
        ▼
search.py                  ← Streamlit UI: keyword search + RAG question answering
```

---

## Setup

**1. Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Create `.env` file**
```
GEMINI_API_KEY=your_key_here
ROOT_DIRECTORY=/path/to/image/folder
MODEL_NAME=models/gemini-2.5-flash
```

---

## Step 1 — OCR: extract text from images

```bash
python extract_text_parallel.py
```

Walks `ROOT_DIRECTORY`, sends every `.jpg` / `.tif` image to Gemini Vision, and writes a sibling `*_ocr.txt` file. Skips images that already have an OCR file. Runs in parallel (configurable via `MAX_WORKERS` in `.env`).

---

## Step 2 — Ingest: load OCR files into SQLite

```bash
python ingest.py
```

Reads all `*_ocr.txt` files under `data/Ackerman/` and writes `letters.db` with:

- **`letters`** table — one row per file, with parsed metadata extracted from the filename
- **`letters_fts`** — FTS5 virtual table for full-text search over body, sender, recipient, and date

Filename convention parsed: `Sender - Recipient - Date[ - Page][ (draft)]`

Re-running is safe; existing rows are skipped (`INSERT OR IGNORE`).

**Schema**

| Column | Description |
|---|---|
| `sender` | Parsed from filename |
| `recipient` | Parsed from filename |
| `date` | Normalised ISO date (`YYYY-MM-DD`, `YYYY-MM`, or `YYYY`) |
| `date_raw` | Date string as it appears in the filename |
| `page` | Page/part number for multi-page letters |
| `is_draft` | 1 if filename contained `(draft)` |
| `subfolder` | Source subfolder (e.g. `From Syracuse`) |
| `body` | Full OCR text |

---

## Step 3 — Search UI

```bash
streamlit run search.py
```

Opens at `http://localhost:8501` with two tabs:

### Search tab
Keyword search with filters for sender, recipient, collection, date range, and drafts. Uses FTS5 directly — supports quoted phrases (`"rocket ship"`), prefix wildcards (`sci*`), and column-scoped queries (`sender:Tucker`).

### Ask tab
RAG question-answering pipeline:
1. Type a natural-language question
2. Click **Generate queries** — Gemini proposes 3 FTS5 search queries
3. Edit any query or add a fourth before searching
4. Click **Search & Answer** — results are retrieved and passed to Gemini as context
5. Answer is shown with source letters expandable below

The **Results per query** slider (1–20) controls how many letters are retrieved per query and therefore how much context is passed to Gemini.
