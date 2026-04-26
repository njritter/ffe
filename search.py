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

DB_PATH = Path("fanzines.db")

st.set_page_config(page_title="FFE Fanzines", layout="wide")
st.title("FFE Fanzines")


@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data
def load_filter_options():
    conn = get_conn()
    fanzines = [r[0] for r in conn.execute(
        "SELECT DISTINCT fanzine FROM pages WHERE fanzine IS NOT NULL ORDER BY fanzine COLLATE NOCASE"
    )]
    provenances = [r[0] for r in conn.execute(
        "SELECT DISTINCT provenance FROM pages WHERE provenance IS NOT NULL ORDER BY provenance"
    )]
    return fanzines, provenances


def sanitize_fts(query: str) -> str:
    """Strip characters that break FTS5 query syntax, keep meaningful words."""
    cleaned = re.sub(r'[^\w\s]', ' ', query)
    tokens = cleaned.split()
    return " ".join(tokens) if tokens else ""


def fts_search(query, limit=10):
    conn = get_conn()
    clean = sanitize_fts(query)
    if not clean:
        return []
    return conn.execute(
        """
        SELECT p.id, p.fanzine, p.issue_folder, p.issue_code, p.volume,
               p.issue_number, p.date, p.date_raw, p.page, p.provenance,
               p.subfolder, p.body,
               bm25(pages_fts) AS score
        FROM pages p
        JOIN pages_fts ON p.id = pages_fts.rowid
        WHERE pages_fts MATCH ?
        ORDER BY bm25(pages_fts)
        LIMIT ?
        """,
        [clean, limit],
    ).fetchall()


def keyword_search(query, fanzine, provenance, date_from, date_to):
    conn = get_conn()
    conditions, params = [], []

    if query:
        clean = sanitize_fts(query)
        if clean:
            conditions.append("p.id IN (SELECT rowid FROM pages_fts WHERE pages_fts MATCH ?)")
            params.append(clean)
    if fanzine:
        conditions.append("p.fanzine = ?")
        params.append(fanzine)
    if provenance:
        conditions.append("p.provenance = ?")
        params.append(provenance)
    if date_from:
        conditions.append("p.date >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("p.date <= ?")
        params.append(date_to)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return conn.execute(
        f"""
        SELECT p.id, p.fanzine, p.issue_folder, p.issue_code, p.volume,
               p.issue_number, p.date, p.date_raw, p.page, p.provenance,
               p.subfolder, p.body
        FROM pages p {where}
        ORDER BY p.fanzine COLLATE NOCASE, p.date ASC NULLS LAST, p.issue_number ASC NULLS LAST, p.page
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


def page_label(row):
    label = row["fanzine"] or "?"
    if row["issue_code"]:
        label += f" — {row['issue_code'].upper()}"
    if row["date"] or row["date_raw"]:
        label += f" ({row['date'] or row['date_raw']})"
    if row["page"]:
        label += f" p.{row['page']}"
    return label


def render_page_card(row, query=""):
    with st.expander(page_label(row)):
        c = st.columns(4)
        c[0].markdown(f"**Fanzine**  \n{row['fanzine'] or '—'}")
        c[1].markdown(f"**Issue**  \n{row['issue_folder'] or '—'}")
        c[2].markdown(f"**Date**  \n{row['date'] or row['date_raw'] or '—'}")
        c[3].markdown(f"**Source**  \n{row['provenance'] or '—'}")
        st.markdown("**Excerpt**")
        st.markdown(highlight(row["body"], query))
        with st.expander("Full text"):
            st.text(row["body"])


# ── Gemini helpers ────────────────────────────────────────────────────────────

def get_db_sample(question: str, n: int = 20) -> list:
    """Quick FTS search to gather context about what's in the DB for this question."""
    conn = get_conn()
    keywords = sanitize_fts(" ".join(re.findall(r'\b\w{4,}\b', question)[:6]))
    if not keywords:
        return []
    try:
        rows = conn.execute(
            """
            SELECT p.fanzine, p.date, p.provenance, p.issue_folder
            FROM pages p
            JOIN pages_fts ON p.id = pages_fts.rowid
            WHERE pages_fts MATCH ?
            ORDER BY bm25(pages_fts)
            LIMIT ?
            """,
            [keywords, n],
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def generate_queries_and_filters(question: str, db_sample: list, fanzines: list, provenances: list) -> dict:
    """Ask Gemini for search queries and filter suggestions based on the question and DB sample."""
    model = genai.GenerativeModel(MODEL_NAME)

    if db_sample:
        fanzines_seen = list(dict.fromkeys(r["fanzine"] for r in db_sample if r["fanzine"]))[:8]
        years_seen = sorted({r["date"][:4] for r in db_sample if r.get("date") and len(r["date"]) >= 4})
        provenances_seen = list(dict.fromkeys(r["provenance"] for r in db_sample if r["provenance"]))[:5]
        sample_text = (
            f"Top matching fanzines: {', '.join(fanzines_seen) or 'none'}\n"
            f"Years seen in results: {', '.join(years_seen) or 'unknown'}\n"
            f"Sources seen: {', '.join(provenances_seen) or 'none'}"
        )
    else:
        sample_text = "No initial results found."

    prompt = f"""You are helping search a database of historical science fiction fanzines from the 1930s–1950s.

The user asked: "{question}"

A quick scan of the database returned:
{sample_text}

Your job is to suggest:
1. Three short FTS5 search queries (2–4 words each, no operators) to retrieve relevant fanzine pages.
2. Optional filters to narrow the search. For fanzine and provenance, the value must exactly match one from the lists below — or use null.
   - Valid fanzines (sample): {', '.join(fanzines_seen[:10]) if db_sample else '(see database)'}
   - Valid provenances (sample): {', '.join(provenances_seen[:5]) if db_sample else '(see database)'}
3. A one-sentence explanation of why you suggested these filters.

Return ONLY valid JSON:
{{
  "queries": ["query1", "query2", "query3"],
  "filters": {{
    "fanzine": null,
    "date_from": null,
    "date_to": null,
    "provenance": null
  }},
  "reasoning": "One sentence explaining the suggestions."
}}"""

    response = model.generate_content(prompt)
    text = response.text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    result = json.loads(text)

    # Validate fanzine/provenance against actual DB values
    f = result.get("filters", {})
    if f.get("fanzine") and f["fanzine"] not in fanzines:
        f["fanzine"] = None
    if f.get("provenance") and f["provenance"] not in provenances:
        f["provenance"] = None

    return result


def answer_question(question: str, pages: list) -> str:
    model = genai.GenerativeModel(MODEL_NAME)

    context_parts = []
    for r in pages:
        header = (
            f"[Page {r['id']}] {r['fanzine']} | "
            f"{r['issue_folder'] or r['issue_code'] or 'unknown issue'} | "
            f"{r['date'] or r['date_raw'] or 'unknown date'} | "
            f"p.{r['page'] or '?'}"
        )
        context_parts.append(f"{header}\n\n{r['body']}")
    context = ("\n\n" + "=" * 60 + "\n\n").join(context_parts)

    prompt = f"""You are a research assistant helping to analyze a collection of historical science fiction fanzines from the 1930s–1950s. These are fan-published magazines from early science fiction fandom.

Answer the question below using ONLY the fanzine pages provided as context. Cite specific pages by referencing the fanzine title, issue, and date. If the pages don't contain enough information to answer fully, say so.

FANZINE PAGES:
{context}

QUESTION: {question}"""

    response = model.generate_content(prompt)
    return response.text


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_search, tab_ask = st.tabs(["Search", "Ask"])

fanzines, provenances = load_filter_options()

# ── Search tab ────────────────────────────────────────────────────────────────
with tab_search:
    with st.sidebar:
        st.header("Search filters")
        query = st.text_input("Search text", placeholder="e.g. rocket ship science fiction")
        fanzine = st.selectbox("Fanzine", [""] + fanzines)
        provenance = st.selectbox("Source collection", [""] + provenances)
        col1, col2 = st.columns(2)
        date_from = col1.text_input("Date from", placeholder="1930")
        date_to = col2.text_input("Date to", placeholder="1950")
        st.caption(f"Database: `{DB_PATH}`")

    if not any([query, fanzine, provenance, date_from, date_to]):
        st.info("Enter a search term or choose a filter in the sidebar.")
    else:
        rows = keyword_search(
            query,
            fanzine or None,
            provenance or None,
            date_from or None,
            date_to or None,
        )
        st.caption(f"{len(rows)} result{'s' if len(rows) != 1 else ''}" +
                   (" (limit 200)" if len(rows) == 200 else ""))
        if not rows:
            st.warning("No pages matched.")
        else:
            for row in rows:
                render_page_card(row, query)

# ── Ask tab ───────────────────────────────────────────────────────────────────
with tab_ask:
    st.markdown("Ask a question about the fanzines. Gemini will search the database, suggest filters, and generate search queries — all of which you can adjust before running.")

    question = st.text_input("Your question", placeholder="e.g. What did fans think about Hugo Gernsback?")
    results_per_query = st.slider("Results per query", min_value=1, max_value=20, value=5,
                                  help="How many pages to retrieve per search query.")

    if question:
        if st.button("Generate queries & suggestions", type="primary"):
            with st.spinner("Scanning database and generating suggestions…"):
                try:
                    db_sample = get_db_sample(question)
                    result = generate_queries_and_filters(question, db_sample, fanzines, provenances)
                    st.session_state.ask_result = result
                    st.session_state.ask_question = question
                except Exception as e:
                    st.error(f"Failed to generate suggestions: {e}")

    if "ask_result" in st.session_state and st.session_state.get("ask_question") == question:
        result = st.session_state.ask_result
        suggested_filters = result.get("filters", {})

        # Reasoning
        if result.get("reasoning"):
            st.info(result["reasoning"])

        st.divider()

        col_queries, col_filters = st.columns([3, 2])

        with col_queries:
            st.markdown("**Search queries**")
            edited_queries = []
            for i, q in enumerate(result.get("queries", [])):
                val = st.text_input(f"Query {i + 1}", value=q, key=f"q_{i}")
                edited_queries.append(val)
            extra = st.text_input("Additional query (optional)", key="q_extra")
            if extra:
                edited_queries.append(extra)

        with col_filters:
            st.markdown("**Suggested filters** — uncheck to remove")

            # Fanzine
            sug_fanzine = suggested_filters.get("fanzine")
            use_fanzine = st.checkbox(
                f"Fanzine: **{sug_fanzine}**" if sug_fanzine else "Fanzine *(none suggested)*",
                value=bool(sug_fanzine),
                disabled=not sug_fanzine,
            )
            ask_fanzine = sug_fanzine if use_fanzine else None

            # Date range
            sug_from = suggested_filters.get("date_from")
            sug_to = suggested_filters.get("date_to")
            date_label = f"Date range: **{sug_from or '?'} – {sug_to or '?'}**" if (sug_from or sug_to) else "Date range *(none suggested)*"
            use_dates = st.checkbox(date_label, value=bool(sug_from or sug_to), disabled=not (sug_from or sug_to))
            if use_dates and (sug_from or sug_to):
                dc1, dc2 = st.columns(2)
                ask_date_from = dc1.text_input("From", value=sug_from or "", key="ask_df")
                ask_date_to = dc2.text_input("To", value=sug_to or "", key="ask_dt")
            else:
                ask_date_from = ask_date_to = None

            # Provenance
            sug_prov = suggested_filters.get("provenance")
            use_prov = st.checkbox(
                f"Source: **{sug_prov}**" if sug_prov else "Source *(none suggested)*",
                value=bool(sug_prov),
                disabled=not sug_prov,
            )
            ask_provenance = sug_prov if use_prov else None

        st.divider()

        if st.button("Search & Answer", type="primary"):
            active_queries = [q for q in edited_queries if q.strip()]

            with st.spinner("Searching fanzines…"):
                seen = {}
                for q in active_queries:
                    try:
                        for row in fts_search(q, limit=results_per_query):
                            if row["id"] not in seen:
                                seen[row["id"]] = row
                    except Exception:
                        pass

                # Apply filters to narrow results
                if any([ask_fanzine, ask_date_from, ask_date_to, ask_provenance]):
                    filtered = {}
                    for rid, row in seen.items():
                        if ask_fanzine and row["fanzine"] != ask_fanzine:
                            continue
                        if ask_date_from and (not row["date"] or row["date"] < ask_date_from):
                            continue
                        if ask_date_to and (not row["date"] or row["date"] > ask_date_to):
                            continue
                        if ask_provenance and row["provenance"] != ask_provenance:
                            continue
                        filtered[rid] = row
                    seen = filtered

            context_pages = list(seen.values())

            if not context_pages:
                st.warning("No relevant pages found after applying filters. Try loosening the filters or adjusting the queries.")
            else:
                with st.spinner(f"Asking Gemini using {len(context_pages)} pages as context…"):
                    try:
                        answer = answer_question(question, context_pages)
                    except Exception as e:
                        st.error(f"Failed to get answer: {e}")
                        st.stop()

                st.markdown("### Answer")
                st.markdown(answer)

                st.markdown(f"### Source pages ({len(context_pages)})")
                for row in context_pages:
                    render_page_card(row)
