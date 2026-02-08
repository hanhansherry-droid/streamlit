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


def is_low_value_title(title: str) -> bool:
    t = (title or "").lower()
    return any(x in t for x in ["disambiguation", "list of", "outline of", "index of"])


def filter_top5_docs(docs, min_words: int = 300):
    """Simple rule-based filtering: no scoring."""
    selected = []
    seen = set()

    for d in docs:
        title = (d.metadata or {}).get("title", "").strip()
        content = (d.page_content or "").strip()

        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())

        if is_low_value_title(title):
            continue

        if word_count(content) < min_words:
            continue

        selected.append(d)
        if len(selected) == 5:
            break

    return selected


def build_context(docs, max_chars: int = 12000) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        text = (d.page_content or "").strip()
        title = (d.metadata or {}).get("title", f"Source {i}")
        parts.append(f"[Source {i}: {title}]\n{text}\n")
    return "\n".join(parts)[:max_chars]


# =========================
# LLM
# =========================
def generate_report(industry: str, context: str) -> str:
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("Missing HF_TOKEN")

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )

    prompt = f"""
You are a market research assistant writing for a consultant-style business analyst.

RULES:
- Use ONLY the information from the Wikipedia sources.
- Cite facts using [Source 1], [Source 2], etc.
- If a section is not supported, say "Not supported by the provided sources."
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
        model="moonshotai/Kimi-K2-Instruct-0905",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1000,
    )

    return res.choices[0].message.content.strip()


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Market Research Assistant", layout="centered")
st.title("Market Research Assistant")
st.write(
    "Retrieve relevant Wikipedia pages for an industry and generate a concise (<500 words) consultant-style summary."
)

with st.form("industry_form"):
    industry = st.text_input(
        "Enter an industry",
        placeholder="e.g. fast fashion, airline industry, semiconductor industry"
    )
    run = st.form_submit_button("Run analysis")

if not run:
    st.info("Enter an industry and click **Run analysis**.")
    st.stop()

industry = industry.strip()
if not industry:
    st.warning("Please enter an industry.")
    st.stop()

# Step 1: Retrieve Wikipedia pages
with st.spinner("Searching Wikipedia..."):
    retriever = WikipediaRetriever(top_k_results=12, lang="en")
    raw_docs = retriever.invoke(industry)

if not raw_docs:
    st.warning("No Wikipedia pages found.")
    st.stop()

# Step 2: Simple filtering
docs = filter_top5_docs(raw_docs)

if not docs:
    st.warning("Pages found but none passed basic screening.")
    st.stop()

st.subheader("Step 2 — Selected Wikipedia pages")
for i, d in enumerate(docs, start=1):
    meta = d.metadata or {}
    st.write(
        f"{i}. **{meta.get('title', 'Untitled')}** "
        f"(Words: {word_count(d.page_content)})"
    )
    st.write(meta.get("source", "URL not available"))

# Step 3: Generate report
if not os.getenv("HF_TOKEN"):
    st.info("HF_TOKEN not set. Report generation disabled.")
    st.stop()

context = build_context(docs)

with st.spinner("Generating industry report..."):
    report = generate_report(industry, context)

st.subheader("Step 3 — Industry report (<500 words)")
st.write(report)
st.caption(f"Word count: {word_count(report)}")
