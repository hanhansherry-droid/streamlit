import os
import re
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from openai import OpenAI

# --- helper: word count ---
def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))

st.title("Market Research Assistant")
st.write("Retrieve relevant Wikipedia pages for a given industry and generate a <500-word report.")

industry = st.text_input(
    "Enter an industry",
    placeholder="e.g. fast fashion, airline industry, semiconductor industry"
)

if st.button("Run (Q1–Q3)"):

    # -------------------------
    # Q1: Validate input
    # -------------------------
    if not industry or not industry.strip():
        st.error("Please provide an industry to continue.")
        st.stop()
    industry = industry.strip()

    # -------------------------
    # Q2: Wikipedia retrieval (top 5)
    # -------------------------
    with st.spinner("Searching Wikipedia..."):
        retriever = WikipediaRetriever(top_k_results=5, lang="en")
        docs = retriever.invoke(industry)

    if not docs:
        st.warning("No relevant Wikipedia pages found.")
        st.stop()

    st.subheader("Step 2 — Top 5 relevant Wikipedia pages (URLs)")
    urls = []
    for doc in docs[:5]:
        url = (doc.metadata or {}).get("source", "URL not available")
        urls.append(url)
        st.write(url)

    # -------------------------
    # Q3: Generate industry report (<500 words) based on these 5 pages
    # -------------------------
    # 1) build context from docs (limit size to avoid huge prompts)
    context_parts = []
    for i, d in enumerate(docs[:5], start=1):
        text = (d.page_content or "").strip()
        if text:
            context_parts.append(f"[Source {i}]\n{text}\n")
    context = "\n".join(context_parts)[:12000]  # limit length for cost/safety

    # 2) init HF Router client (Kimi)
    if not os.getenv("HF_TOKEN"):
        st.error("Missing HF_TOKEN environment variable. Please set it and rerun.")
        st.stop()

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )

    # 3) prompt
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

    with st.spinner("Generating industry report (<500 words)..."):
        completion = client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct-0905",
            messages=[{"role": "user", "content": prompt}],
        )

    report = completion.choices[0].message.content.strip()

    # 4) enforce <500 words (most reliable: compress again if needed)
    if word_count(report) > 500:
        compress_prompt = f"""
Compress the report below to UNDER 500 words.
Keep it factual and keep the [Source #] citations.
Do not add new information.

REPORT:
{report}
"""
        completion2 = client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct-0905",
            messages=[{"role": "user", "content": compress_prompt}],
        )
        report = completion2.choices[0].message.content.strip()

    st.subheader("Step 3 — Industry report (<500 words)")
    st.write(report)
    st.caption(f"Word count: {word_count(report)} (must be < 500)")

