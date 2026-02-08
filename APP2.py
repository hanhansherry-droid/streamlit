import os
import re
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from openai import OpenAI


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def extract_max_year(text: str) -> int:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", text or "")
    years_int = [int(y) for y in years if 1900 <= int(y) <= 2100]
    return max(years_int) if years_int else 0


def is_low_value_title(title: str) -> bool:
    t = (title or "").lower()
    return any(x in t for x in ["disambiguation", "list of", "outline of", "index of"])


def select_5_docs(docs, min_words: int = 300, preferred_year: int = 2022):
    """
    Must return 5 docs (if possible from candidates).
    Hard filters (must pass):
      - not disambiguation/list/outline/index
      - not duplicate title
      - min word count
    Soft preference (not required):
      - max_year >= preferred_year
    """
    hard_pass = []
    seen = set()

    # 1) hard filter
    for d in docs:
        meta = d.metadata or {}
        title = (meta.get("title") or "Untitled").strip()
        content = (d.page_content or "").strip()

        key = title.lower()
        if key in seen:
            continue
        seen.add(key)

        if is_low_value_title(title):
            continue

        wc = word_count(content)
        if wc < min_words:
            continue

        max_year = extract_max_year(content)
        url = meta.get("source", "URL not available")

        hard_pass.append(
            {
                "doc": d,
                "title": title,
                "url": url,
                "words": wc,
                "max_year": max_year,
            }
        )

    # if too few after hard filter, just return what we have
    if len(hard_pass) <= 5:
        selected = hard_pass
    else:
        # 2) soft preference: recent first, then others
        preferred = [x for x in hard_pass if x["max_year"] >= preferred_year]
        fallback = [x for x in hard_pass if x["max_year"] < preferred_year]

        selected = preferred[:5]
        if len(selected) < 5:
            selected += fallback[: (5 - len(selected))]

    # add note for display
    for x in selected:
        x["note"] = "Preferred (recent)" if x["max_year"] >= preferred_year else "Fallback"
    return selected[:5]


def build_context(selected, max_chars: int = 12000) -> str:
    parts = []
    for i, item in enumerate(selected, start=1):
        text = (item["doc"].page_content or "").strip()
        title = item["title"]
        parts.append(f"[Source {i}: {title}]\n{text}\n")
    return "\n".join(parts)[:max_chars]


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
- Use ONLY the information in the Wikipedia sources.
- Cite facts using [Source 1], [Source 2], etc.
- If a section is not supported, say: "Not supported by the provided sources."
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
# Streamlit UI
# =========================
st.set_page_config(page_title="Market Research Assistant", layout="centered")
st.title("Market Research Assistant")
st.write("Retrieve Wikipedia pages, apply simple screening (must output 5), then generate a <500-word report based ONLY on selected sources.")

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

# Step 1: retrieve candidates
with st.spinner("Searching Wikipedia..."):
    retriever = WikipediaRetriever(top_k_results=12, lang="en")
    raw_docs = retriever.invoke(industry)

if not raw_docs:
    st.warning("No relevant Wikipedia pages found. Try a different industry keyword.")
    st.stop()

# Step 2: select 5 with hard filters + soft preference
selected = select_5_docs(raw_docs, min_words=300, preferred_year=2022)

if not selected:
    st.warning("No pages passed the basic filters. Try a different keyword (e.g., add 'industry').")
    st.stop()

# If fewer than 5 exist after hard filtering, warn but still show
if len(selected) < 5:
    st.warning(f"Only {len(selected)} pages passed basic screening. Showing all available pages.")

st.subheader("Step 2 — Selected Wikipedia pages (Title | Words | MaxYear | Note | URL)")
for i, item in enumerate(selected, start=1):
    year_display = item["max_year"] if item["max_year"] else "N/A"
    st.write(
        f"{i}. **{item['title']}** | Words: **{item['words']}** | "
        f"MaxYear: **{year_display}** | **{item['note']}**"
    )
    st.write(item["url"])

# Step 3: generate report
if not os.getenv("HF_TOKEN"):
    st.info("Report generation is disabled because HF_TOKEN is not set. Set HF_TOKEN to enable it.")
    st.stop()

context = build_context(selected)

with st.spinner("Generating industry report (<500 words)..."):
    report = generate_report_kimi(industry, context)

st.subheader("Step 3 — Industry report (<500 words)")
st.write(report)
st.caption(f"Word count: {word_count(report)}")
