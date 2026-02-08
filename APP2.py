import os
import re
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from openai import OpenAI
from dataclasses import dataclass
from typing import List, Tuple

# =========================
# Data structure
# =========================
@dataclass
class WikiPageInfo:
    title: str
    url: str
    content: str
    score: float
    year_max: int
    citations_count: int
    word_count: int
    flags: List[str]


# =========================
# Helpers
# =========================
def wc(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))

def count_citations(text: str) -> int:
    # Wikipedia often has [1], [2] style refs in extracted text
    return len(re.findall(r"\[\d+\]", text or ""))

def extract_max_year(text: str) -> int:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text or "")
    years_int = [int(y) for y in years if 1900 <= int(y) <= 2100]
    return max(years_int) if years_int else 0

def is_bad_title(title: str) -> bool:
    t = (title or "").lower()
    bad_patterns = [
        "disambiguation",
        "list of",
        "outline of",
        "index of",
    ]
    return any(p in t for p in bad_patterns)

def relevance_score(industry: str, title: str, content: str) -> float:
    # very simple relevance proxy: industry keywords appear in title/content
    ind = (industry or "").lower().strip()
    t = (title or "").lower()
    c = (content or "").lower()

    if not ind:
        return 0.0

    # split into keywords (avoid super strict NLP)
    keywords = [k for k in re.split(r"[\s,;/\-]+", ind) if len(k) >= 3]

    in_title = sum(1 for k in keywords if k in t)
    in_content = sum(1 for k in keywords if k in c[:3000])  # focus on early content

    # normalize
    title_part = min(in_title / max(len(keywords), 1), 1.0)
    content_part = min(in_content / max(len(keywords), 1), 1.0)
    return 0.6 * title_part + 0.4 * content_part

def page_quality_score(industry: str, title: str, content: str) -> Tuple[float, List[str], int, int, int]:
    """
    Score = weighted heuristic:
    - length (depth)
    - citations (reference density)
    - recency (mentions recent years)
    - relevance (industry match)
    Plus flags for transparency.
    """
    flags = []
    if is_bad_title(title):
        flags.append("List/Disambiguation-like title")

    words = wc(content)
    cits = count_citations(content)
    max_year = extract_max_year(content)

    # depth score (consulting-style: avoid stubs)
    depth = min(words / 1200, 1.0)  # 1200 words ~ decent baseline
    if words < 600:
        flags.append("Very short content (possible stub)")
    elif words < 1200:
        flags.append("Short content (may lack depth)")

    # citations score
    cit_score = min(cits / 15, 1.0)  # 15+ gives full credit
    if cits < 3:
        flags.append("Few citations/refs in extracted text")

    # recency score (your requirement: prefer recent years)
    # Wikipedia may not always include recent years, so this is a soft preference
    if max_year >= 2023:
        recency = 1.0
    elif max_year >= 2020:
        recency = 0.7
    elif max_year >= 2015:
        recency = 0.4
    else:
        recency = 0.2
        flags.append("No clear recent year mentions")

    # relevance
    rel = relevance_score(industry, title, content)
    if rel < 0.25:
        flags.append("Weak industry keyword match (possible off-topic)")

    # additional red flags
    if "citation needed" in (content or "").lower():
        flags.append("Contains 'citation needed'")

    # total weighted score
    score = (
        0.30 * rel +
        0.25 * depth +
        0.25 * cit_score +
        0.20 * recency
    )

    return score, flags, max_year, cits, words

def evaluate_and_select_top5(industry: str, docs) -> List[WikiPageInfo]:
    evaluated: List[WikiPageInfo] = []

    for d in docs:
        meta = d.metadata or {}
        title = meta.get("title", "").strip() or "Untitled"
        url = meta.get("source", "URL not available")
        content = (d.page_content or "").strip()

        score, flags, year_max, cits, words = page_quality_score(industry, title, content)

        # hard filters (your “必须筛查”的部分)
        # 1) remove obvious list/disambiguation (unless we have too few later)
        hard_bad = is_bad_title(title)
        # 2) avoid extremely short pages
        too_short = words < 300
        # 3) avoid very weak relevance
        too_irrelevant = relevance_score(industry, title, content) < 0.15

        if hard_bad:
            flags.append("Filtered: low reference value (list/disambiguation)")
        if too_short:
            flags.append("Filtered: too short to support analysis")
        if too_irrelevant:
            flags.append("Filtered: not clearly about the industry")

        evaluated.append(WikiPageInfo(
            title=title,
            url=url,
            content=content,
            score=score,
            year_max=year_max,
            citations_count=cits,
            word_count=words,
            flags=flags
        ))

    # filter out hard-bad first
    filtered = [p for p in evaluated if "Filtered:" not in " ".join(p.flags)]
    # If filtered too aggressively, fall back to best-scoring pages
    if len(filtered) < 5:
        filtered = sorted(evaluated, key=lambda x: x.score, reverse=True)

    # sort by score and take top 5
    filtered = sorted(filtered, key=lambda x: x.score, reverse=True)[:5]
    return filtered

def build_context(pages: List[WikiPageInfo], max_chars: int = 12000) -> str:
    parts = []
    for i, p in enumerate(pages, start=1):
        if p.content:
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

    # show sources list in prompt (helps grounding)
    sources_list = "\n".join([f"- [Source {i}] {p.title}" for i, p in enumerate(pages, start=1)])

    prompt = f"""
You are a market research assistant writing for a business analyst at a large corporation.

STRICT RULES (must follow):
- Use ONLY the information in the provided Wikipedia sources.
- Do NOT use external knowledge.
- If a section cannot be supported, write: "Information not available in the provided sources."
- Cite key statements using [Source 1], [Source 2], etc.
- Keep the full report UNDER 500 words.

Industry: {industry}

The five sources are:
{sources_list}

Write a concise, consultant-style report with these headings:

## Industry overview
(Definition, scope, and what the industry includes/excludes.)

## Industry development trends
(How the industry has evolved and what directional changes are described in the sources. If "latest news" is not in sources, say so.)

## Competitive landscape
(Key players/types of players, concentration vs fragmentation, and competitive dynamics described in sources.)

## Customers and demand drivers
(Customer types/segments and what drives demand, only if supported.)

## Risks and challenges
(Regulatory, reputational, sustainability, operational risks, controversies, etc.)

## Information gaps (next research priorities)
(What important data or facts are missing from Wikipedia sources and what a real analyst should research next.)

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
# Streamlit UI (simple, no sidebar, no example buttons)
# =========================
st.set_page_config(page_title="Market Research Assistant", layout="centered")
st.title("Market Research Assistant")
st.write("Enter an industry. The app retrieves the most useful Wikipedia pages and generates a <500-word consultant-style summary based only on those sources.")

industry = st.text_input("Enter an industry", placeholder="e.g., fast fashion, airline industry, semiconductor industry")

# Q1: input validation
if not industry or not industry.strip():
    st.info("Please enter an industry to begin.")
    st.stop()

industry = industry.strip()

# Q2: retrieve & screen pages
with st.spinner("Searching Wikipedia..."):
    retriever = WikipediaRetriever(top_k_results=12, lang="en")  # fetch more, then screen
    raw_docs = retriever.invoke(industry)

if not raw_docs:
    st.warning("No relevant Wikipedia pages found. Try a different industry keyword.")
    st.stop()

pages = evaluate_and_select_top5(industry, raw_docs)

st.subheader("Step 2 — Screened top 5 Wikipedia pages (URLs + quality notes)")
for i, p in enumerate(pages, start=1):
    st.write(f"{i}. {p.url}")
    st.caption(
        f"Title: {p.title} | Score: {p.score:.2f} | Max year: {p.year_max or 'N/A'} | "
        f"Words: {p.word_count} | Citations: {p.citations_count}"
    )
    # show only the most important notes (avoid clutter)
    if p.flags:
        # keep flags short
        short_flags = p.flags[:3]
        st.caption("Notes: " + " | ".join(short_flags))

# Q3: report generation
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

