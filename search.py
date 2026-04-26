import json
import re
import sqlite3
from pathlib import Path

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = os.getenv("MODEL_NAME", "models/gemini-2.5-flash-preview-04-17")

DB_PATH = Path("letters.db")

st.set_page_config(page_title="FFE Letters", layout="wide")
st.title("FFE Letters")


@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data
def load_filter_options():
    conn = get_conn()
    senders = [r[0] for r in conn.execute(
        "SELECT DISTINCT sender FROM letters WHERE sender IS NOT NULL ORDER BY sender"
    )]
    recipients = [r[0] for r in conn.execute(
        "SELECT DISTINCT recipient FROM letters WHERE recipient IS NOT NULL ORDER BY recipient"
    )]
    subfolders = [r[0] for r in conn.execute(
        "SELECT DISTINCT subfolder FROM letters WHERE subfolder IS NOT NULL ORDER BY subfolder"
    )]
    return senders, recipients, subfolders


def fts_search(query, limit=10):
    conn = get_conn()
    return conn.execute(
        """
        SELECT l.id, l.filename, l.sender, l.recipient, l.date, l.date_raw,
               l.page, l.is_draft, l.subfolder, l.body,
               bm25(letters_fts) AS score
        FROM letters l
        JOIN letters_fts ON l.id = letters_fts.rowid
        WHERE letters_fts MATCH ?
        ORDER BY bm25(letters_fts)
        LIMIT ?
        """,
        [query, limit],
    ).fetchall()


def keyword_search(query, sender, recipient, subfolder, date_from, date_to, drafts_only):
    conn = get_conn()
    conditions, params = [], []

    if query:
        conditions.append("l.id IN (SELECT rowid FROM letters_fts WHERE letters_fts MATCH ?)")
        params.append(query)
    if sender:
        conditions.append("l.sender = ?")
        params.append(sender)
    if recipient:
        conditions.append("l.recipient = ?")
        params.append(recipient)
    if subfolder:
        conditions.append("l.subfolder = ?")
        params.append(subfolder)
    if date_from:
        conditions.append("l.date >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("l.date <= ?")
        params.append(date_to)
    if drafts_only:
        conditions.append("l.is_draft = 1")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return conn.execute(
        f"""
        SELECT l.id, l.filename, l.sender, l.recipient, l.date, l.date_raw,
               l.page, l.is_draft, l.subfolder, l.body
        FROM letters l {where}
        ORDER BY l.date ASC NULLS LAST, l.filename
        LIMIT 200
        """,
        params,
    ).fetchall()


def highlight(text, query, max_chars=300):
    if not query or not text:
        return text[:max_chars] + ("…" if len(text) > max_chars else "")
    terms = re.findall(r"\w+", query.lower())
    lower = text.lower()
    first = min((lower.find(t) for t in terms if lower.find(t) >= 0), default=0)
    start = max(0, first - 80)
    excerpt = ("…" if start > 0 else "") + text[start: start + max_chars]
    if start + max_chars < len(text):
        excerpt += "…"
    for term in terms:
        excerpt = re.sub(f"({re.escape(term)})", r"**\1**", excerpt, flags=re.IGNORECASE)
    return excerpt


def letter_label(row):
    parts = [row["sender"] or "?", "→", row["recipient"] or "?"]
    parts.append(f"({row['date'] or row['date_raw'] or '—'})")
    if row["is_draft"]:
        parts.append("*draft*")
    return " ".join(parts)


def render_letter_card(row, query=""):
    with st.expander(letter_label(row)):
        c = st.columns(4)
        c[0].markdown(f"**Sender**  \n{row['sender'] or '—'}")
        c[1].markdown(f"**Recipient**  \n{row['recipient'] or '—'}")
        c[2].markdown(f"**Date**  \n{row['date'] or row['date_raw'] or '—'}")
        c[3].markdown(f"**Collection**  \n{row['subfolder'] or 'Root'}")
        st.markdown("**Excerpt**")
        st.markdown(highlight(row["body"], query))
        with st.expander("Full text"):
            st.text(row["body"])


# ── Gemini helpers ────────────────────────────────────────────────────────────

def generate_queries(question: str) -> list:
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = f"""You are helping search a database of historical science fiction fan letters from the 1930s–1950s.
The database uses SQLite FTS5 full-text search.

Generate 3 short FTS5 search queries that together would retrieve the letters most relevant to answering the question below.
Each query should focus on different keywords or angles. Keep queries short (2–4 words).
Avoid stop words. Do not use FTS5 syntax operators.

Return ONLY a JSON array of strings, e.g. ["query one", "query two", "query three"].

Question: {question}"""

    response = model.generate_content(prompt)
    text = response.text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def answer_question(question: str, letters: list) -> str:
    model = genai.GenerativeModel(MODEL_NAME)

    context_parts = []
    for r in letters:
        header = f"[Letter {r['id']}] {r['sender']} → {r['recipient']} | {r['date'] or r['date_raw'] or 'unknown date'}"
        context_parts.append(f"{header}\n\n{r['body']}")
    context = "\n\n{'='*60}\n\n".join(context_parts)

    prompt = f"""You are a research assistant helping to analyze a collection of historical science fiction fan letters from the 1930s–1950s, primarily correspondence involving Forrest J Ackerman and his circle.

Answer the question below using ONLY the letters provided as context. Cite specific letters by referencing the sender, recipient, and date. If the letters don't contain enough information to answer fully, say so.

LETTERS:
{context}

QUESTION: {question}"""

    response = model.generate_content(prompt)
    return response.text


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_search, tab_ask = st.tabs(["Search", "Ask"])

# ── Search tab ────────────────────────────────────────────────────────────────
with tab_search:
    senders, recipients, subfolders = load_filter_options()

    with st.sidebar:
        st.header("Search filters")
        query = st.text_input("Search text", placeholder="e.g. rocket ship science fiction")
        sender = st.selectbox("Sender", [""] + senders)
        recipient = st.selectbox("Recipient", [""] + recipients)
        subfolder = st.selectbox("Collection", [""] + subfolders)
        col1, col2 = st.columns(2)
        date_from = col1.text_input("Date from", placeholder="1930")
        date_to = col2.text_input("Date to", placeholder="1950")
        drafts_only = st.checkbox("Drafts only")
        st.caption(f"Database: `{DB_PATH}`")

    if not any([query, sender, recipient, subfolder, date_from, date_to, drafts_only]):
        st.info("Enter a search term or choose a filter in the sidebar.")
    else:
        rows = keyword_search(
            query, sender or None, recipient or None, subfolder or None,
            date_from or None, date_to or None, drafts_only,
        )
        st.caption(f"{len(rows)} result{'s' if len(rows) != 1 else ''}" +
                   (" (limit 200)" if len(rows) == 200 else ""))
        if not rows:
            st.warning("No letters matched.")
        else:
            for row in rows:
                render_letter_card(row, query)

# ── Ask tab ───────────────────────────────────────────────────────────────────
with tab_ask:
    st.markdown("Ask a question about the letters. Gemini will generate search queries which you can edit before running.")

    question = st.text_input("Your question", placeholder="e.g. What did people think about Hugo Gernsback?")
    results_per_query = st.slider("Results per query", min_value=1, max_value=20, value=5,
                                  help="How many letters to retrieve per search query. Higher values give more context but cost more tokens.")

    # Generate queries button — only re-calls Gemini when question changes
    if question:
        if st.button("Generate queries", type="primary"):
            with st.spinner("Generating search queries…"):
                try:
                    st.session_state.generated_queries = generate_queries(question)
                    st.session_state.queries_for_question = question
                except Exception as e:
                    st.error(f"Failed to generate queries: {e}")

    # Show editable queries once generated
    if "generated_queries" in st.session_state and st.session_state.get("queries_for_question") == question:
        st.markdown("**Edit queries, then click Search & Answer:**")

        edited_queries = []
        for i, q in enumerate(st.session_state.generated_queries):
            val = st.text_input(f"Query {i + 1}", value=q, key=f"q_{i}")
            edited_queries.append(val)

        extra = st.text_input("Additional query (optional)", placeholder="e.g. fanzine publication", key="q_extra")
        if extra:
            edited_queries.append(extra)

        if st.button("Search & Answer", type="primary"):
            active_queries = [q for q in edited_queries if q.strip()]

            with st.spinner("Searching letters…"):
                seen = {}
                for q in active_queries:
                    try:
                        for row in fts_search(q, limit=results_per_query):
                            if row["id"] not in seen:
                                seen[row["id"]] = row
                    except Exception:
                        pass  # bad FTS query, skip

            context_letters = list(seen.values())

            if not context_letters:
                st.warning("No relevant letters found. Try adjusting the queries.")
            else:
                with st.spinner(f"Asking Gemini using {len(context_letters)} letters as context…"):
                    try:
                        answer = answer_question(question, context_letters)
                    except Exception as e:
                        st.error(f"Failed to get answer: {e}")
                        st.stop()

                st.markdown("### Answer")
                st.markdown(answer)

                st.markdown(f"### Source letters ({len(context_letters)})")
                for row in context_letters:
                    render_letter_card(row)
