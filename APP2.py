import os
import re
import time
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from openai import OpenAI
from dataclasses import dataclass
from typing import List


# =========================
# Data structure
# =========================
@dataclass
class WikiPageInfo:
    title: str
    url: str
    content: str
    max_year: int
    word_count: int


# =========================
# Helpers
# =========================
def wc(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def extract_max_year(text: str) -> int:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text or "")
    years_int = [int(y) for y in years if 1900 <= int(y) <= 2100]
    return max(years_int) if years_int else 0


def is_low_value_page(title: str) -> bool:
    """Filter pages that are usually not good industry sources."""
    t = (title or "").lower()
    bad_patterns = ["disambiguation", "list of", "outline of", "index of"]
    return any(p in t for p in bad_patterns)


def pick_top5_docs(industry: str, docs, min_words: int = 300) -> List[WikiPageInfo]:
    """
    Selection criteria:
    1) Most relevant: keep retriever order as primary relevance signal.
    2) Prefer recent: use max_year as tie-break among candidates.
    3) Filter out disambiguation/list/outline.
    4) Filter out very short pages (word_count < min_words).
    5) De-duplicate by title.
    """
    candidates: List[WikiPageInfo] = []
    seen_titles = set()

    # Step A: basic filtering while preserving original order
    for d in docs:
        meta = d.metadata or {}
        title = (meta.get("title") or "").strip() or "Untitled"
        url = meta.get("source", "URL not available")
        content = (d.page_content or "").strip()

        if title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())

        if is_low_value_page(title):
            continue

        words = wc(content)
        if words < min_words:
            continue

        candidates.append(
            WikiPageInfo(
                title=title,
                url=url,
                content=content,
                max_year=extract_max_year(content),
                word_count=words,
            )
        )

    # If filtering is too strict, fall back to the first usable pages (still no scoring)
    if len(candidates) < 5:
        fallback = []
        seen_titles = set()
        for d in docs:
            meta = d.metadata or {}
            title = (meta.get("title") or "").strip() or "Untitled"
            url = meta.get("source", "URL not available")
            content = (d.page_content or "").strip()

            if title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())

            fallback.append(
                WikiPageInfo(
                    title=title,
                    url=url,
                    content=content,
                    max_year=extract_max_year(content),
                    word_count=wc(content),
                )
            )
            if len(fallback) == 5:
                break
        return fallback

    # Step B (light preference for recency): sort candidates by max_year DESC,
    # but keep relevance by only applying this within the already "most relevant set".
    # To avoid breaking relevance too much, we take the first 8 in relevance order,
    # then sort those by year and take top 5.
    top_by_relevance = candidates[:8]  # relevance proxy: original order
    top_by_relevance.sort(key=lambda x: x.max_year, reverse=True)

    return top_by_relevance[:5]


def build_context(pages: List[WikiPageInfo], max_chars: int = 12000) -> str:
    parts = []
    for i, p in enumerate(pages, start=1):
        parts.append(f"[Source {i}: {p.title}]\n{p.content}\n")
    ctx = "\n".join(parts)
    return ctx[:max_chars]


def generate_report(industry: str, context: str, pages: List[WikiPageInfo]) -> str:
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("Missing HF_TOKEN. Set it in Streamlit Cloud → Secrets.")

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )

    sources_list = "\n".join([f"- [Source {i}] {p.title}" for i, p in enumerate(pages, start=1)])

    prompt = f"""
You are a market research assistant writing for a business analyst at a large corporation.

RULES:
- Use ONLY the information from the provided Wikipedia sources.
- Do NOT use external knowledge.
- Cite key statements using [Source 1], [Source 2], etc.
- Keep the full report UNDER 500 words.
- Write in a neutral, analytical, consultant-style tone.

Industry: {industry}

The five sources are:
{sources_list}

Write a concise report with these headings:

## Industry overview
(Definition and scope.)

## Industry development trends
(How the industry has evolved and major directional changes described in the sources.)

## Competitive landscape
(Key players/types of players and competition dynamics described.)

## Customers and demand characteristics
(Customer groups and demand drivers if supported.)

## Risks and challenges
(Regulatory, operational, reputational, sustainability challenges mentioned.)

Sources:
{context}
"""

    completion = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct-0905",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1100,
    )
    return completion.choices[0].message.content.strip()


def compress_if_needed(industry: str, report: str) -> str:
    if wc(report) <= 500:
        return report

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )

    prompt = f"""
Compress the report below to UNDER 500 words.
Keep headings and [Source #] citations.
Do not add any new information.

Industry: {industry}

REPORT:
{report}
"""
    completion = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct-0905",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=900,
    )
    return completion.choices[0].message.content.strip()


# =========================
# Streamlit UI (simple + stable reruns)
# =========================
st.set_page_config(page_title="Market Research Assistant", layout="centered")
st.title("Market Research Assistant")
st.write("Enter an industry. The app retrieves the top 5 relevant Wikipedia pages and generates a <500-word report based only on those sources.")

industry = st.text_input("Enter an industry", placeholder="e.g., fast fashion, airline industry, semiconductor industry")

# Debounce to avoid rerun on every keystroke
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "last_time" not in st.session_state:
    st.session_state.last_time = 0.0

now = time.time()
if industry != st.session_state.last_input:
    st.session_state.last_input = industry
    st.session_state.last_time = now
    st.stop()

if now - st.session_state.last_time < 0.6:
    st.stop()

# Q1
if not industry or not industry.strip():
    st.info("Please enter an industry to begin.")
    st.stop()

industry = industry.strip()

# Q2
with st.spinner("Searching Wikipedia..."):
    retriever = WikipediaRetriever(top_k_results=12, lang="en")
    raw_docs = retriever.invoke(industry)

if not raw_docs:
    st.warning("No relevant Wikipedia pages found. Try a different industry keyword.")
    st.stop()

pages = pick_top5_docs(industry, raw_docs, min_words=300)

st.subheader("Step 2 — Top 5 relevant Wikipedia pages (Title | Year | Word count | URL)")
for i, p in enumerate(pages, start=1):
    year_display = p.max_year if p.max_year else "N/A"
    st.write(f"{i}. **{p.title}** | Year: **{year_display}** | Words: **{p.word_count}**")
    st.write(p.url)

# Q3
if not os.getenv("HF_TOKEN"):
    st.info("Q3 (report generation) is disabled because HF_TOKEN is not set in Streamlit Cloud Secrets.")
    st.stop()

context = build_context(pages)

with st.spinner("Generating industry report (<500 words)..."):
    try:
        report = generate_report(industry, context, pages)
    except Exception as e:
        st.error("LLM call failed. Debug info:")
        st.write(type(e).__name__)
        st.write(str(e))
        st.stop()

report = compress_if_needed(industry, report)

st.subheader("Step 3 — Industry report (<500 words)")
st.write(report)
st.caption(f"Word count: {wc(report)} (must be < 500)")
