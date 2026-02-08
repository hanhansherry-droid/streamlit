import os
import re
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from openai import OpenAI


# =========================
# Helpers
# =========================
def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def extract_max_year(text: str) -> int:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text or "")
    years_int = [int(y) for y in years if 1900 <= int(y) <= 2100]
    return max(years_int) if years_int else 0


def is_low_value_title(title: str) -> bool:
    t = (title or "").lower()
    return any(x in t for x in ["disambiguation", "list of", "outline of", "index of"])


def screen_docs(docs, min_words: int = 300, min_year: int = 2022):
    """
    Simple rule-based screening (NO scoring):
    - remove disambiguation/list/outline/index pages
    - dedupe by title
    - require minimum content length
    - require recent year mentioned in content (>= min_year)
    Returns:
      selected: list of dicts for top 5
      log: list of (title, reason) for transparency
    """
    selected = []
    log = []
    seen = set()

    for d in docs:
        meta = d.metadata or {}
        title = (meta.get("title") or "Untitled").strip()
        url = meta.get("source", "URL not available")
        content = (d.page_content or "").strip()

        key = title.lower()
        if key in seen:
            log.append((title, "Skipped: duplicate title"))
            continue
        seen.add(key)

        if is_low_value_title(title):
            log.append((title, "Rejected: disambiguation/list/outline/index page"))
            continue

        wc = word_count(content)
        if wc < min_words:
            log.append((title, f"Rejected: too short (<{min_words} words)"))
            continue

        max_year = extract_max_year(content)
        if max_year and max_year < min_year:
            log.append((title, f"Rejected: no recent years (max_year={max_year} < {min_year})"))
            continue
        if max_year == 0:
            # If no year found, we keep it as a fallback candidate
            log.append((title, "Accepted (fallback): no year detected, but content length ok"))
        else:
            log.append((title, f"Accepted: recent years detected (max_year={max_year})"))

        selected.append(
            {
                "doc": d,
                "title": title,
                "url": url,
                "words": wc,
                "max_year": max_year,
                "note": "Selected",
            }
        )

        if len(selected) == 5:
            break

    return selected, log


def build_context(selected, max_chars: int = 12000) -> str:
    parts = []
    for i, item in enumerate(selected, start=1):
        d = item["doc"]
        text = (d.page_content or "").strip()
        title = item["title"]
        parts.append(f"[Source {i}: {title}]\n{text}\n")
    return "\n".join(parts)[:max_chars]


# =========================
# LLM
# =========================
def generate_report_kimi(industry: str, context: str) -> str:
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("Missing HF_TOKEN environment variable.")

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )

    prompt = f"""
You are a market research assistant for a consultant-style business analyst.

RULES:
- Use ONLY the information from the Wikipedia sources.
- Cite facts using [Source 1], [Source 2], etc.
- If a section is not supported, write: "Not supported by the provided sources."
- Keep the report UNDER 500 words.

Industry: {industry}

Write the report with EXACT headings:

## Industry overview
## Competitive landscape
## Customers and demand characteristics
## Technology and innovation (if supported)
## Recent developments mentioned in the sources
## Risks and challenges

Sources:
{context}
"""

    completion = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct-0905",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1000,
    )

    return completion.choices[0].message.content.strip()


# =========================
# Streamlit UI (Run button)
# =========================
st.set_page_config(page_title="Market Research Assistant", layout="centered")
st.title("Market Research Assistant")
st.write("Retrieve Wikipedia pages, apply simple screening rules, then generate a <500-word report based ONLY on selected sources.")

with st.form("industry_form"):
    industry = st.text_input(
        "Enter an industry",
        placeholder="e.g. fast fashion, airline industry, semiconductor industry"
    )
    run = st.form_submit_button("Run analysis")

if not run:
    st.info("Enter an industry and click **Run analysis**.")
    st.stop()

industry = (industry or "").strip()
if not industry:
    st.warning("Please enter an industry.")
    st.stop()

# Step 1: Retrieve more than 5, then screen down to 5
with st.spinner("Searching Wikipedia..."):
    retriever = WikipediaRetriever(top_k_results=12, lang="en")
    raw_docs = retriever.invoke(industry)

if not raw_docs:
    st.warning("No relevant Wikipedia pages found. Try a different industry keyword.")
    st.stop()

selected, log = screen_docs(raw_docs, min_words=300, min_year=2022)

if not selected:
    st.warning("Pages found but none passed screening. Try a broader industry keyword (e.g., add 'industry').")
    st.stop()

# Step 2: Show selected sources + simple transparency log
st.subheader("Step 2 — Selected Wikipedia sources (Title | MaxYear | Words | URL)")
for i, item in enumerate(selected, start=1):
    year_display = item["max_year"] if item["max_year"] else "N/A"
    st.write(f"{i}. **{item['title']}** | MaxYear: **{year_display}** | Words: **{item['words']}**")
    st.write(item["url"])

with st.expander("Screening log (why pages were accepted/rejected)"):
    for title, reason in log[:30]:
        st.write(f"- **{title}** — {reason}")
    if len(log) > 30:
        st.write(f"... ({len(log)-30} more)")

# Step 3: Generate report
if not os.getenv("HF_TOKEN"):
    st.info("Report generation is disabled because HF_TOKEN is not set. Set HF_TOKEN to enable it.")
    st.stop()

context = build_context(selected)

with st.spinner("Generating industry report (<500 words)..."):
    report = generate_report_kimi(industry, context)

st.subheader("Step 3 — Industry report (<500 words)")
st.write(report)
st.caption(f"Word count: {word_count(report)}")
