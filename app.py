import os
import re
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from openai import OpenAI


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def build_context(docs, max_chars: int = 12000) -> str:
    """Build a bounded context string from the top 5 Wikipedia docs."""
    parts = []
    for i, d in enumerate(docs[:5], start=1):
        text = (d.page_content or "").strip()
        if text:
            parts.append(f"[Source {i}]\n{text}\n")
    context = "\n".join(parts)
    return context[:max_chars]


def generate_report_kimi(industry: str, context: str) -> str:
    """Call HF Router (Kimi) to generate an industry report."""
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("Missing HF_TOKEN environment variable.")

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )

    prompt = f"""
You are a market research assistant for a business analyst at a large corporation.

Write an industry report UNDER 500 words.
Use ONLY the information in the sources below (Wikipedia pages).
If the sources do not contain an answer, say so.
When stating facts, cite sources like [Source 1], [Source 2], etc.

Industry: {industry}

Sources:
{context}
"""

    completion = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct-0905",
        messages=[{"role": "user", "content": prompt}],
    )

    return completion.choices[0].message.content.strip()


def compress_to_500_words_if_needed(industry: str, report: str) -> str:
    """If report is >500 words, ask the model to compress it (no new info)."""
    if not os.getenv("HF_TOKEN"):
        return report  # can't compress without token; return as-is

    if word_count(report) <= 500:
        return report

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )

    compress_prompt = f"""
Compress the report below to UNDER 500 words.
Keep it factual and keep the [Source #] citations.
Do not add new information.

Industry: {industry}

REPORT:
{report}
"""

    completion2 = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct-0905",
        messages=[{"role": "user", "content": compress_prompt}],
    )

    return completion2.choices[0].message.content.strip()


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Market Research Assistant", layout="centered")
st.title("Market Research Assistant")
st.write("Type an industry to automatically retrieve Wikipedia pages and generate a short report (<500 words).")

industry = st.text_input(
    "Enter an industry",
    placeholder="e.g. fast fashion, airline industry, semiconductor industry"
)

# -------------------------
# Auto-run (no button)
# -------------------------
if industry and industry.strip():
    industry = industry.strip()

    # Q2: Wikipedia retrieval
    with st.spinner("Searching Wikipedia..."):
        retriever = WikipediaRetriever(top_k_results=5, lang="en")
        docs = retriever.invoke(industry)

    if not docs:
        st.warning("No relevant Wikipedia pages found. Try a different industry keyword.")
        st.stop()

    st.subheader("Step 2 — Top 5 relevant Wikipedia pages (URLs)")
    for doc in docs[:5]:
        url = (doc.metadata or {}).get("source", "URL not available")
        st.write(url)

    # Q3: Generate report (only if HF_TOKEN exists)
    if not os.getenv("HF_TOKEN"):
        st.info("Q3 (report generation) is disabled because HF_TOKEN is not set. Set HF_TOKEN to enable it.")
        st.stop()

    context = build_context(docs)

    # ✅ try/except 放在这里：包住真正的 LLM 调用
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

else:
    # Q1: No industry provided
    st.info("Please enter an industry to begin.")
