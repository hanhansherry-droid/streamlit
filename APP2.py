import os
import re
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from openai import OpenAI

# -------------------------
# Helpers
# -------------------------
def wc(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))

def is_bad_title(title: str) -> bool:
    t = (title or "").lower()
    bad_patterns = [
        "disambiguation",
        "list of",
        "outline of",
    ]
    return any(p in t for p in bad_patterns)

def basic_quality_flags(title: str, content: str) -> list[str]:
    flags = []
    if is_bad_title(title):
        flags.append("May be a list/disambiguation page")
    if len(content or "") < 6000:  # simple proxy for depth
        flags.append("Short page (may lack depth)")
    if "citation needed" in (content or "").lower():
        flags.append("Contains 'citation needed'")
    return flags

def build_context(docs, max_chars: int = 12000) -> str:
    parts = []
    for i, d in enumerate(docs[:5], start=1):
        text = (d.page_content or "").strip()
        title = (d.metadata or {}).get("title", f"Source {i}")
        if text:
            parts.append(f"[Source {i}: {title}]\n{text}\n")
    ctx = "\n".join(parts)
    return ctx[:max_chars]

def generate_report(industry: str, context: str) -> str:
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("Missing HF_TOKEN. Set it in Streamlit Cloud → Secrets.")

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )

    prompt = f"""
You are a market research assistant writing for a business analyst at a large corporation.

TASK:
Write an industry report UNDER 500 words based ONLY on the Wikipedia sources provided.
If a section cannot be supported by the sources, write "Information not available in the provided sources."

RULES:
- Use ONLY the text in Sources.
- No outside knowledge.
- Cite sources for key claims using [Source 1], [Source 2], etc.
- Keep the report concise and structured.

REPORT STRUCTURE (use these headings):
1) Industry overview
2) Competitive landscape
3) Technology and innovation
4) Customers and demand drivers
5) Risks and challenges
6) Information gaps (what Wikipedia did not cover well)

Industry: {industry}

Sources:
{context}
"""

    completion = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct-0905",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1200,
    )
    return completion.choices[0].message.content.strip()

def compress_if_needed(industry: str, report: str) -> str:
    if wc(report) <= 500:
        return report
    # If it's too long, compress (still no new info)
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

# -------------------------
# Streamlit UI (simple)
# -------------------------
st.set_page_config(page_title="Market Research Assistant", layout="centered")
st.title("Market Research Assistant")
st.write("Enter an industry. The app retrieves the top 5 Wikipedia pages and generates a <500-word report based only on those sources.")

industry = st.text_input("Enter an industry", placeholder="e.g., fast fashion, airline industry, semiconductor industry")

# Q1: check input
if not industry or not industry.strip():
    st.info("Please enter an industry to begin.")
    st.stop()

industry = industry.strip()

# Q2: retrieve pages
with st.spinner("Searching Wikipedia..."):
    retriever = WikipediaRetriever(top_k_results=10, lang="en")  # fetch more then keep best 5
    raw_docs = retriever.invoke(industry)

if not raw_docs:
    st.warning("No relevant Wikipedia pages found. Try a different industry keyword.")
    st.stop()

# simple filtering: remove disambiguation/list pages first, then take first 5
filtered = []
for d in raw_docs:
    title = (d.metadata or {}).get("title", "")
    if not is_bad_title(title):
        filtered.append(d)

docs = (filtered if len(filtered) >= 5 else raw_docs)[:5]

st.subheader("Step 2 — Top 5 relevant Wikipedia pages (URLs)")
for i, doc in enumerate(docs, start=1):
    meta = doc.metadata or {}
    url = meta.get("source", "URL not available")
    title = meta.get("title", f"Source {i}")
    flags = basic_quality_flags(title, doc.page_content or "")
    st.write(f"{i}. {url}")
    if flags:
        st.caption("Quality notes: " + " | ".join(flags))

# Q3: report generation
if not os.getenv("HF_TOKEN"):
    st.info("Q3 (report generation) is disabled because HF_TOKEN is not set in Streamlit Cloud Secrets.")
    st.stop()

context = build_context(docs)

with st.spinner("Generating industry report (<500 words)..."):
    try:
        report = generate_report(industry, context)
    except Exception as e:
        st.error("LLM call failed. Debug info:")
        st.write(type(e).__name__)
        st.write(str(e))
        st.stop()

report = compress_if_needed(industry, report)

st.subheader("Step 3 — Industry report (<500 words)")
st.write(report)
st.caption(f"Word count: {wc(report)} (must be < 500)")
