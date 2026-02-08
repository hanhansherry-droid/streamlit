import os
import re
import streamlit as st
import requests
from dataclasses import dataclass
from typing import List, Dict, Any
from langchain_community.retrievers import WikipediaRetriever
from openai import OpenAI

WIKI_API = "https://en.wikipedia.org/w/api.php"
UA = "MarketResearchAssistant/1.0 (student project)"


# =========================
# Helpers
# =========================
def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def extract_max_year(text: str) -> int:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text or "")
    years_int = [int(y) for y in years if 1900 <= int(y) <= 2100]
    return max(years_int) if years_int else 0


def count_citation_needed(text: str) -> int:
    return len(re.findall(r"citation needed", (text or "").lower()))


def is_low_value_title(title: str) -> bool:
    t = (title or "").lower()
    bad = ["disambiguation", "list of", "outline of", "index of"]
    return any(x in t for x in bad)


def is_valid_url(url: str, timeout: float = 4.0) -> bool:
    """Basic URL availability check."""
    if not url or not url.startswith("http"):
        return False
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout, headers={"User-Agent": UA})
        if r.status_code >= 400:
            r = requests.get(url, allow_redirects=True, timeout=timeout, headers={"User-Agent": UA})
        return r.status_code < 400
    except Exception:
        return False


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def fetch_wiki_signals(title: str) -> Dict[str, Any]:
    """
    Wikipedia-only signals (no scoring):
    - last edit year (revision timestamp)
    - categories (for rough company-like detection)
    - disambiguation marker
    """
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "revisions|categories|pageprops",
        "rvprop": "timestamp",
        "cllimit": "max",
    }
    try:
        resp = requests.get(WIKI_API, params=params, timeout=6, headers={"User-Agent": UA})
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {}) if pages else {}

        revs = page.get("revisions", []) or []
        ts = revs[0].get("timestamp", "") if revs else ""
        last_edit_year = int(ts[:4]) if ts[:4].isdigit() else 0

        cats = page.get("categories", []) or []
        categories = [c.get("title", "") for c in cats if isinstance(c, dict)]

        pageprops = page.get("pageprops", {}) or {}
        is_disambig = "disambiguation" in pageprops

        return {"last_edit_year": last_edit_year, "categories": categories, "is_disambig": is_disambig}
    except Exception:
        return {"last_edit_year": 0, "categories": [], "is_disambig": False}


def is_company_like(title: str, categories: List[str]) -> bool:
    """
    Heuristic: label company-like pages to enforce concept-vs-company balance.
    """
    t = (title or "").lower()
    if any(k in t for k in ["inc.", "ltd", "plc", "corp", "company", "group"]):
        return True
    cat_text = " ".join([c.lower() for c in (categories or [])])
    markers = [
        "companies", "brands", "retailers", "manufacturers", "airlines",
        "software companies", "technology companies", "organizations"
    ]
    return any(m in cat_text for m in markers)


@dataclass
class PageInfo:
    title: str
    url: str
    content: str
    words: int
    max_year: int
    last_edit_year: int
    citation_needed: int
    is_company: bool
    note: str  # why selected / fallback reason


def screen_top5_pages(
    docs,
    *,
    min_words: int = 300,
    min_content_year: int = 2022,
    min_last_edit_year: int = 2022,
    max_citation_needed: int = 3,
    min_non_company: int = 2,
) -> List[PageInfo]:
    """
    Rule-based screening (NO scoring):
    - remove disambiguation/list/outline/index
    - deduplicate titles
    - require min words
    - require url reachable (if exists)
    - require recency: content max_year >= min_content_year AND last_edit_year >= min_last_edit_year
      (if too strict -> fallback pool used)
    - limit 'citation needed'
    - enforce diversity: try to include min_non_company concept pages in final top5
    """
    passed: List[PageInfo] = []
    fallback: List[PageInfo] = []
    seen = set()

    for d in docs:
        meta = d.metadata or {}
        title = (meta.get("title") or "").strip() or "Untitled"
        url = (meta.get("source") or "").strip() or "URL not available"
        content = (d.page_content or "").strip()

        key = title.lower()
        if key in seen:
            continue
        seen.add(key)

        # basic filters
        if is_low_value_title(title):
            continue

        words = word_count(content)
        if words < min_words:
            continue

        if url.startswith("http") and not is_valid_url(url):
            continue

        sig = fetch_wiki_signals(title)
        if sig.get("is_disambig", False):
            continue

        max_year = extract_max_year(content)
        last_edit_year = sig.get("last_edit_year", 0)
        cit_need = count_citation_needed(content)
        company_like = is_company_like(title, sig.get("categories", []))

        # strict pass conditions
        strict_ok = True
        reasons = []

        if max_year < min_content_year:
            strict_ok = False
            reasons.append(f"Fallback: content may be older (max_year={max_year or 'N/A'}).")
        else:
            reasons.append(f"Recent years mentioned (max_year={max_year}).")

        if last_edit_year < min_last_edit_year:
            strict_ok = False
            reasons.append(f"Fallback: last edit year {last_edit_year} (<{min_last_edit_year}).")
        else:
            reasons.append(f"Recently maintained (last_edit_year={last_edit_year}).")

        if cit_need > max_citation_needed:
            strict_ok = False
            reasons.append(f"Fallback: too many 'citation needed' ({cit_need}).")
        else:
            reasons.append(f"'citation needed' acceptable ({cit_need}).")

        page = PageInfo(
            title=title,
            url=url,
            content=content,
            words=words,
            max_year=max_year,
            last_edit_year=last_edit_year,
            citation_needed=cit_need,
            is_company=company_like,
            note=" | ".join(reasons),
        )

        if strict_ok:
            passed.append(page)
        else:
            fallback.append(page)

    # Enforce diversity in final 5 (if possible)
    selected: List[PageInfo] = []

    non_company = [p for p in passed if not p.is_company]
    company = [p for p in passed if p.is_company]

    selected.extend(non_company[:min_non_company])

    def add_until_5(pool: List[PageInfo]):
        titles = {p.title.lower() for p in selected}
        for p in pool:
            if len(selected) == 5:
                break
            if p.title.lower() in titles:
                continue
            selected.append(p)
            titles.add(p.title.lower())

    # Fill remaining from strict-passed first
    add_until_5(passed)

    # If still <5, relax using fallback
    if len(selected) < 5:
        add_until_5(fallback)

    return selected[:5]


def build_context(pages: List[PageInfo], max_chars: int = 12000) -> str:
    parts = []
    for i, p in enumerate(pages, start=1):
        parts.append(f"[Source {i}: {p.title}]\n{p.content}\n")
    context = "\n".join(parts)
    return context[:max_chars]


# =========================
# LLM calls
# =========================
def generate_report_kimi(industry: str, context: str) -> str:
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("Missing HF_TOKEN environment variable.")

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )

    prompt = f"""
You are a market research assistant writing for a consultant-style business analyst.

STRICT RULES:
- Use ONLY the information in the provided Wikipedia sources.
- Every section must include at least one citation like [Source 1].
- If a section is not supported by the sources, write: "Not supported by the provided sources."
- Keep the full report UNDER 500 words.
- Neutral, analytical tone.

Industry: {industry}

Write the report with EXACTLY these headings:

## Industry overview
(Definition and scope.)

## Competitive landscape
(Key players/types of players and competition dynamics mentioned.)

## Customers and demand characteristics
(Customer groups, demand drivers, adoption patterns if supported.)

## Technology and innovation (only if supported)
(Technologies/standards/innovations explicitly mentioned in sources.)

## Recent developments mentioned in the sources
(Only developments/events explicitly stated in sources; do not invent news beyond Wikipedia.)

## Risks and challenges
(Regulatory, operational, reputational, sustainability issues mentioned.)

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


def compress_to_500_words_if_needed(industry: str, report: str) -> str:
    if not os.getenv("HF_TOKEN"):
        return report

    if word_count(report) <= 500:
        return report

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )

    compress_prompt = f"""
Compress the report below to UNDER 500 words.
Keep the headings and keep the [Source #] citations.
Do not add new information.

Industry: {industry}

REPORT:
{report}
"""

    completion2 = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct-0905",
        messages=[{"role": "user", "content": compress_prompt}],
        temperature=0.1,
        max_tokens=900,
    )

    return completion2.choices[0].message.content.strip()


# =========================
# Streamlit UI (with Run button)
# =========================
st.set_page_config(page_title="Market Research Assistant", layout="centered")
st.title("Market Research Assistant")
st.write(
    "Enter an industry. Click **Run analysis** to retrieve screened Wikipedia pages and generate a <500-word consultant-style summary based only on those sources."
)

with st.form("industry_form", clear_on_submit=False):
    industry = st.text_input(
        "Enter an industry",
        placeholder="e.g. fast fashion, airline industry, semiconductor industry"
    )
    run = st.form_submit_button("Run analysis")

if not run:
    st.info("Enter an industry and click **Run analysis**.")
    st.stop()

if not industry or not industry.strip():
    st.warning("Please enter an industry keyword before running.")
    st.stop()

industry = industry.strip()

# Step 2: Retrieve more, then screen to top 5
with st.spinner("Searching Wikipedia (top 12) and screening pages..."):
    retriever = WikipediaRetriever(top_k_results=12, lang="en")
    raw_docs = retriever.invoke(industry)

if not raw_docs:
    st.warning("No relevant Wikipedia pages found. Try a different industry keyword.")
    st.stop()

pages = screen_top5_pages(
    raw_docs,
    min_words=300,
    min_content_year=2022,
    min_last_edit_year=2022,
    max_citation_needed=3,
    min_non_company=2,
)

if not pages:
    st.warning("Pages were found but none passed screening. Try a broader industry keyword.")
    st.stop()

st.subheader("Step 2 — Screened Top 5 Wikipedia pages (Title | URL | Words | Recency | Notes)")
for i, p in enumerate(pages, start=1):
    st.write(
        f"{i}. **{p.title}** | "
        f"Words: **{p.words}** | MaxYear: **{p.max_year or 'N/A'}** | "
        f"LastEdit: **{p.last_edit_year or 'N/A'}** | "
        f"Type: **{'Company' if p.is_company else 'Industry/Concept'}**"
    )
    st.write(p.url)
    st.caption(p.note)

# Step 3: Generate report
if not os.getenv("HF_TOKEN"):
    st.info("Q3 (report generation) is disabled because HF_TOKEN is not set. Set HF_TOKEN to enable it.")
    st.stop()

context = build_context(pages)

with st.spinner("Generating industry report (<500 words)..."):
    try:
        report = generate_report_kimi(industry, context)
    except Exception as e:
        st.error("LLM call failed. Showing debug info:")
        st.write(type(e).__name__)
        st.write(str(e))
        st.stop()

report = compress_to_500_words_if_needed(industry, report)

st.subheader("Step 3 — Industry report (<500 words)")
st.write(report)
st.caption(f"Word count: {word_count(report)} (must be < 500)")
