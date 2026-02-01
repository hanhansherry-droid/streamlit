import streamlit as st
from langchain_community.retrievers import WikipediaRetriever

st.title("Market Research Assistant")
st.write("Retrieve relevant Wikipedia pages for a given industry.")

industry = st.text_input(
    "Enter an industry",
    placeholder="e.g. fast fashion, airline industry, semiconductor industry"
)

if st.button("Retrieve Wikipedia pages"):
    if not industry or not industry.strip():
        st.error("Please provide an industry to continue.")
        st.stop()

    industry = industry.strip()

    with st.spinner("Searching Wikipedia..."):
        retriever = WikipediaRetriever(top_k_results=5, lang="en")
        docs = retriever.invoke(industry)   

    if not docs:
        st.warning("No relevant Wikipedia pages found.")
        st.stop()

    st.subheader("Top 5 relevant Wikipedia pages")
    for doc in docs[:5]:
        url = doc.metadata.get("source", "URL not available")
        st.write(url)

