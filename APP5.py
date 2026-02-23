import re
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from openai import OpenAI


# Helpers
def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))

def extract_max_year(text: str) -> int:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text or "")
    years_int = [int(y) for y in years if 1900 <= int(y) <= 2100]
    return max(years_int) if years_int else 0

def is_low_value_title(title: str) -> bool:
    t = (title or "").lower()
    return any(x in t for x in ["disambiguation", "list of", "outline of", "index of"])

def select_top5_docs(docs, min_words: int = 300, preferred_year: int = 2022):
    """
    Simple rule-based selection (no scoring).
    Hard rules:
      - exclude disambiguation/list/outline/index pages
      - deduplicate by title
      - minimum word count
    Soft preference (not required):
      - prefer pages with max_year >= preferred_year
    Must return 5 pages if possible (fallback to non-screened top results if needed).
    """
    candidates = []
    seen = set()

    # Hard screening
    for d in docs:
        meta = d.metadata or {}
        title = (meta.get("title") or "").strip()
        content = (d.page_content or "").strip()

        if not title:
            continue

        key = title.lower()
        if key in seen:
            continue
        seen.add(key)

        if is_low_value_title(title):
            continue

        wc = word_count(content)
        if wc < min_words:
            continue

        candidates.append({
            "doc": d,
            "title": title,
            "url": meta.get("source", "URL not available"),
            "words": wc,
            "max_year": extract_max_year(content),
        })

    # 2) Fallback to ensure 5 outputs
    if len(candidates) < 5:
        fallback = []
        seen = set()
        for d in docs:
            meta = d.metadata or {}
            title = (meta.get("title") or "").strip()
            if not title:
                continue
            key = title.lower()
            if key in seen:
                continue
            seen.add(key)

            content = (d.page_content or "").strip()
            fallback.append({
                "doc": d,
                "title": title,
                "url": meta.get("source", "URL not available"),
                "words": word_count(content),
                "max_year": extract_max_year(content),
                "note": "Fallback (not enough pages passed screening)",
            })
            if len(fallback) == 5:
                break
        return fallback

    # 3) Soft preference: recent first, then fill
    preferred = [x for x in candidates if x["max_year"] >= preferred_year]
    others = [x for x in candidates if x["max_year"] < preferred_year]

    selected = preferred[:5]
    if len(selected) < 5:
        selected += others[: (5 - len(selected))]

    for x in selected:
        x["note"] = "Preferred (recent)" if x["max_year"] >= preferred_year else "Selected"
    return selected[:5]

def build_context(selected, max_chars: int = 12000) -> str:
    parts = []
    for i, item in enumerate(selected, start=1):
        text = (item["doc"].page_content or "").strip()
        parts.append(f"[Source {i}: {item['title']}]\n{text}\n")
    return "\n".join(parts)[:max_chars]

# LLM
def generate_report(industry: str, context: str, api_key: str, model: str) -> str:
    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=api_key)

    prompt = f"""
You are a market research assistant writing for a consultant-style business analyst.

RULES:
- Use ONLY the information from the Wikipedia sources.
- Cite facts using [Source 1], [Source 2], etc.
- If a section is not supported, write: "Not supported by the provided sources."
- Keep the report UNDER 500 words.

Industry: {industry}

Write the report with these sections:

## Industry overview
## Competitive landscape
## Customers and demand characteristics
## Technology and innovation (if supported)
## Recent developments mentioned in the sources
## Risks and challenges

Sources:
{context}
"""
    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1000,
    )
    return res.choices[0].message.content.strip()

def compress_report_if_needed(industry: str, report: str, api_key: str, model: str, max_words: int = 500) -> str:
    """Enforce <500 words (Q3). No new info allowed."""
    if word_count(report) <= max_words:
        return report

    client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=api_key)

    compress_prompt = f"""
Compress the report below to UNDER {max_words} words.
Keep the headings and [Source #] citations.
Do not add any new information.

Industry: {industry}

REPORT:
{report}
"""
    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": compress_prompt}],
        temperature=0.1,
        max_tokens=900,
    )
    return res.choices[0].message.content.strip()


# Streamlit UI
st.set_page_config(page_title="Market Research Assistant", layout="centered")
st.title("Market Research Assistant")
st.write("Enter an industry → retrieve 5 Wikipedia pages → generate a <500-word report based only on those pages.")

# ---- Sidebar (Q0 requirement) ----
st.sidebar.header("LLM Settings")

selected_model = st.sidebar.selectbox(
    "Select LLM",
    ["moonshotai/Kimi-K2-Instruct-0905"]  # final version: only ONE model
)
api_key = st.sidebar.text_input("Enter your API key", type="password")

st.sidebar.header("Source Screening")
min_words = st.sidebar.slider("Minimum words per page", 100, 800, 300, 50)
preferred_year = st.sidebar.selectbox("Prefer pages mentioning year >=", [2018, 2020, 2022, 2023, 2024], index=2)

# ---- Run button ----
with st.form("industry_form"):
    industry = st.text_input("Enter an industry", placeholder="e.g. fashion, fast food, AI")
    run = st.form_submit_button("Run analysis")

if not run:
    st.info("Enter an industry and click **Run analysis**.")
    st.stop()

industry = (industry or "").strip()
if not industry:
    st.warning("Please enter an industry.")
    st.stop()

if not api_key:
    st.warning("Please enter your API key in the sidebar (markers will use their own).")
    st.stop()

# Step 2: Retrieve candidates
with st.spinner("Searching Wikipedia..."):
    retriever = WikipediaRetriever(top_k_results=12, lang="en")
    raw_docs = retriever.invoke(industry)

if not raw_docs:
    st.warning("No Wikipedia pages found. Try a different industry keyword.")
    st.stop()

# Step 2: Select top 5
selected = select_top5_docs(raw_docs, min_words=min_words, preferred_year=preferred_year)

st.subheader("Step 2 — Top 5 Wikipedia pages (Title | Words | MaxYear | Note | URL)")
for i, item in enumerate(selected, start=1):
    year_display = item["max_year"] if item["max_year"] else "N/A"
    st.write(f"{i}. **{item['title']}** | Words: **{item['words']}** | MaxYear: **{year_display}** | **{item['note']}**")
    st.write(item["url"])

# Step 3: Generate report
context = build_context(selected)

with st.spinner("Generating industry report (<500 words)..."):
    report = generate_report(industry, context, api_key, selected_model)
    report = compress_report_if_needed(industry, report, api_key, selected_model, max_words=500)

st.subheader("Step 3 — Industry report (<500 words)")
st.write(report)
st.caption(f"Word count: {word_count(report)} (must be < 500)")
