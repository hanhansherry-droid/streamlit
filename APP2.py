import os
import re
import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from openai import OpenAI
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class WikiPageInfo:
    """Data class for Wikipedia page information"""
    title: str
    url: str
    content: str
    reliability_score: float = 0.0
    last_updated: str = "Unknown"
    word_count: int = 0
    issues: List[str] = None
    citations_count: int = 0
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        self.word_count = len(re.findall(r"\b\w+\b", self.content))
        # Count citations in content
        self.citations_count = len(re.findall(r'\[\d+\]', self.content))

def word_count(text: str) -> int:
    """Calculate word count of text"""
    return len(re.findall(r"\b\w+\b", text))

def evaluate_wiki_page(doc, index: int) -> WikiPageInfo:
    """Evaluate Wikipedia page for reference value and calculate reliability score"""
    content = doc.page_content.strip()
    metadata = doc.metadata or {}
    url = metadata.get("source", "URL not available")
    title = metadata.get("title", f"Document {index}")
    
    # Initialize page info
    page_info = WikiPageInfo(
        title=title,
        url=url,
        content=content
    )
    
    # Calculate reliability score (based on heuristic rules)
    content_length_score = min(len(content) / 2000, 1.0)  # 2000 chars as baseline
    
    # Check for citations/references
    citations_score = min(page_info.citations_count / 10, 1.0)  # 10 citations as full score
    
    # Check for recent information
    recent_info_score = 0.3 if any(year in content for year in ["2023", "2024", "2025"]) else 0.1
    
    # Check for comprehensive structure
    structure_score = 0.0
    structure_keywords = ["history", "background", "industry", "market", "technology", "competition"]
    found_keywords = sum(1 for keyword in structure_keywords if keyword in content.lower())
    structure_score = min(found_keywords / 3, 1.0)
    
    # Calculate total score (weighted)
    page_info.reliability_score = (
        content_length_score * 0.3 +
        citations_score * 0.3 +
        recent_info_score * 0.2 +
        structure_score * 0.2
    )
    
    # Check for potential issues
    if len(content) < 1000:
        page_info.issues.append("Short content (may lack depth)")
    if page_info.citations_count < 3:
        page_info.issues.append("Limited citations")
    if "disambiguation" in title.lower():
        page_info.issues.append("Disambiguation page - may need more specific topic")
    if "citation needed" in content.lower():
        page_info.issues.append("Contains unverified claims")
    if "stub" in content.lower()[:500]:
        page_info.issues.append("May be a stub/article in development")
    
    return page_info

def filter_and_rank_docs(docs, min_reliability: float = 0.4) -> List[WikiPageInfo]:
    """Filter and rank documents based on reliability"""
    evaluated_pages = []
    
    for i, doc in enumerate(docs, 1):
        page_info = evaluate_wiki_page(doc, i)
        evaluated_pages.append(page_info)
    
    # Sort by reliability score
    evaluated_pages.sort(key=lambda x: x.reliability_score, reverse=True)
    
    # Filter out low-reliability pages
    filtered_pages = [p for p in evaluated_pages if p.reliability_score >= min_reliability]
    
    # If too few pages after filtering, return top 5 regardless
    if len(filtered_pages) < 3 and evaluated_pages:
        return evaluated_pages[:5]
    
    return filtered_pages[:5]

def build_context_with_metadata(pages: List[WikiPageInfo], max_chars: int = 12000) -> Tuple[str, str]:
    """Build context string with metadata"""
    parts = []
    metadata_parts = []
    
    for i, page in enumerate(pages[:5], start=1):
        # Main content
        text = page.content.strip()
        if text:
            parts.append(f"[Source {i}: {page.title}]\n{text}\n")
        
        # Metadata
        metadata_parts.append(
            f"Source {i}: '{page.title}' | "
            f"Reliability Score: {page.reliability_score:.2f}/1.0 | "
            f"Word Count: {page.word_count} | "
            f"Citations: {page.citations_count} | "
            f"Issues: {', '.join(page.issues) if page.issues else 'None'}"
        )
    
    context = "\n".join(parts)
    metadata = "\n".join(metadata_parts)
    
    return context[:max_chars], metadata

def generate_consultant_report(industry: str, context: str, metadata: str) -> str:
    """Generate professional industry report for consultants"""
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("Missing HF_TOKEN environment variable.")

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )

    prompt = f"""You are a senior consultant at a top-tier consulting firm (McKinsey, BCG, or Bain). Your task is to write a professional industry analysis using ONLY the provided Wikipedia sources.

CRITICAL INSTRUCTIONS:
1. Use ONLY information from the provided Wikipedia sources. DO NOT use any external knowledge.
2. If information for a section is not available in sources, state: "Information not available in provided sources."
3. Cite specific sources for each key fact using [Source X] notation.
4. Highlight contradictions or information gaps when they exist.
5. Maintain a professional, analytical tone suitable for business strategy.

SOURCE QUALITY ASSESSMENT (for your reference only):
{metadata}

REQUIRED REPORT STRUCTURE (use Markdown headings exactly as shown):

## 1. Executive Summary
- Key industry characteristics and scope
- Current state and major trends
- Critical uncertainties

## 2. Industry Overview & Ecosystem
- Industry definition and segmentation
- Historical development timeline
- Market structure and value chain
- Key industry associations and regulatory bodies

## 3. Competitive Landscape Analysis
- Major players and market positioning
- Market share dynamics (if available)
- Barriers to entry
- Competitive advantages/disadvantages

## 4. Technology & Innovation Assessment
- Current technology infrastructure
- Emerging technologies and R&D focus
- Technology adoption curves
- Intellectual property landscape

## 5. Policy & Regulatory Environment
- Key regulations and compliance requirements
- Government initiatives and subsidies
- International standards and agreements
- Regulatory risks and opportunities

## 6. Market Dynamics & Customer Analysis
- Demand drivers and growth trends
- Customer segmentation
- Geographic market variations
- Pricing dynamics and business models

## 7. Risk Assessment Matrix
- Market and competitive risks
- Regulatory and compliance risks
- Technology disruption risks
- Supply chain and operational risks
- Macroeconomic risks

## 8. Strategic Implications & Recommendations
- Key success factors for industry players
- Critical uncertainties to monitor
- Strategic positioning opportunities
- Potential partnership or M&A considerations

## 9. Information Gaps & Research Priorities
- Specific information missing from sources
- Recommended areas for primary research
- Key questions requiring further investigation

INDUSTRY TO ANALYZE: {industry}

WIKIPEDIA SOURCES:
{context}

IMPORTANT: End with this exact disclaimer:
---
**Disclaimer**: This analysis is based solely on publicly available Wikipedia content as of {st.session_state.get('analysis_date', 'current date')}. Wikipedia may contain inaccuracies or outdated information. For strategic business decisions, consult primary sources, conduct market research, and consider expert opinions. This analysis should be treated as a starting point for further investigation, not as a definitive business guide.
"""

    completion = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct-0905",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2500
    )

    return completion.choices[0].message.content.strip()

def compress_report_if_needed(industry: str, report: str) -> str:
    """Compress report while maintaining structure"""
    if not os.getenv("HF_TOKEN"):
        return report

    if word_count(report) <= 1000:  # Increased word limit for comprehensive analysis
        return report

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )

    compress_prompt = f"""Compress the following consultant report to UNDER 1000 words while:
1. Keeping ALL section headings intact
2. Preserving ALL source citations [Source X]
3. Maintaining critical analysis and risk assessments
4. Prioritizing quantitative data and key insights
5. Keeping the disclaimer unchanged
6. Ensuring no new information is added

Industry: {industry}

REPORT TO COMPRESS:
{report}"""

    completion = client.chat.completions.create(
        model="moonshotai/Kimi-K2-Instruct-0905",
        messages=[{"role": "user", "content": compress_prompt}],
        temperature=0.1
    )

    return completion.choices[0].message.content.strip()

# =========================
# Streamlit UI
# =========================
st.set_page_config(
    page_title="Consultant-Grade Industry Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar configuration
with st.sidebar:
    st.title(" Analysis Settings")
    
    st.markdown("### Source Quality Filters")
    min_reliability = st.slider(
        "Minimum Source Reliability",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.1,
        help="Filter out Wikipedia pages with reliability scores below this threshold"
    )
    
    st.markdown("### Display Options")
    show_source_details = st.checkbox("Show Detailed Source Analysis", value=True)
    show_confidence_indicators = st.checkbox("Show Confidence Indicators", value=True)
    
    st.divider()
    
    st.markdown("### About")
    st.info("""
    **Professional Industry Analysis Tool**
    
    This tool provides consultant-grade industry analysis based on Wikipedia sources. 
    
    **Limitations:**
    - Analysis limited to Wikipedia content
    - May miss recent developments
    - Should be supplemented with primary research
    
    **Best for:** Initial market assessment, hypothesis generation, research planning.
    """)
    
    st.caption(f"Version 2.0 | Updated: {datetime.now().strftime('%Y-%m-%d')}")

# Main content area
st.title(" Consultant-Grade Industry Analysis")
st.markdown("""
Generate professional industry reports using Wikipedia as a starting point. 
Optimized for business consultants, strategists, and market researchers.
""")

# Initialize session state for analysis date
if 'analysis_date' not in st.session_state:
    st.session_state.analysis_date = datetime.now().strftime("%B %d, %Y")

# Industry input
col1, col2 = st.columns([3, 1])
with col1:
    industry = st.text_input(
        "**Enter industry for analysis**",
        placeholder="e.g., renewable energy, pharmaceutical industry, e-commerce logistics",
        help="Be specific for better results. Include modifiers like 'emerging markets' or 'digital transformation in [industry]'"
    )

with col2:
    analysis_scope = st.selectbox(
        "Analysis Scope",
        ["Comprehensive", "Quick Overview", "Deep Dive"],
        help="Choose the depth of analysis"
    )

# Main analysis workflow
if industry and industry.strip():
    industry = industry.strip()
    
    # Store in session for report generation
    st.session_state.current_industry = industry
    
    # Step 1: Search Wikipedia
    with st.status(" Searching and evaluating Wikipedia sources...", expanded=True) as status:
        try:
            # Retrieve more docs initially for filtering
            retriever = WikipediaRetriever(top_k_results=10, lang="en")
            raw_docs = retriever.invoke(industry)
            
            if not raw_docs:
                st.error(" No relevant Wikipedia pages found. Please try different keywords.")
                st.stop()
            
            # Filter and rank pages
            evaluated_pages = filter_and_rank_docs(raw_docs, min_reliability)
            
            if not evaluated_pages:
                st.error("No sources meet the minimum reliability criteria. Try lowering the threshold or using different keywords.")
                st.stop()
            
            status.update(
                label=f"Found {len(evaluated_pages)} qualified sources",
                state="complete",
                expanded=False
            )
            
        except Exception as e:
            st.error(f"Search failed: {str(e)}")
            st.stop()
    
    # Step 2: Display source information
    st.subheader(" Source Quality Assessment")
    
    if show_source_details:
        # Create metrics row
        avg_reliability = sum(p.reliability_score for p in evaluated_pages) / len(evaluated_pages)
        total_citations = sum(p.citations_count for p in evaluated_pages)
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Average Reliability Score", f"{avg_reliability:.2f}/1.0")
        with m2:
            st.metric("Total Citations", total_citations)
        with m3:
            st.metric("Total Content", f"{sum(p.word_count for p in evaluated_pages):,} words")
        
        # Display detailed source cards
        for idx, page in enumerate(evaluated_pages):
            with st.expander(f"Source {idx+1}: {page.title}", expanded=False):
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    st.write(f"**URL:** {page.url}")
                    st.write(f"**Content Preview:** {page.content[:200]}...")
                
                with col_right:
                    # Reliability gauge
                    score = page.reliability_score
                    if score >= 0.7:
                        st.success(f"**Reliability:** {score:.2f}/1.0")
                    elif score >= 0.4:
                        st.warning(f"**Reliability:** {score:.2f}/1.0")
                    else:
                        st.error(f"**Reliability:** {score:.2f}/1.0")
                    
                    st.write(f"**Word Count:** {page.word_count}")
                    st.write(f"**Citations:** {page.citations_count}")
                    
                    if page.issues:
                        st.warning("**Issues:** " + ", ".join(page.issues))
    
    # Step 3: Generate report
    st.subheader(" Generating Consultant Report")
    
    if not os.getenv("HF_TOKEN"):
        st.error(" HF_TOKEN environment variable not set. Report generation disabled.")
        st.info("Set HF_TOKEN to enable AI-powered report generation.")
        st.stop()
    
    # Build context with metadata
    context, metadata = build_context_with_metadata(evaluated_pages)
    
    # Adjust max tokens based on analysis scope
    if analysis_scope == "Quick Overview":
        max_tokens = 1500
    elif analysis_scope == "Deep Dive":
        max_tokens = 3000
    else:
        max_tokens = 2500
    
    # Generate report
    with st.spinner(f"Generating {analysis_scope.lower()} report..."):
        try:
            report = generate_consultant_report(industry, context, metadata)
            
            # Compress if needed
            if word_count(report) > 1000 and analysis_scope != "Deep Dive":
                report = compress_report_if_needed(industry, report)
            
            # Display report with tabs
            tab1, tab2 = st.tabs([" Full Report", " Source Context"])
            
            with tab1:
                st.markdown(report)
                
                # Report metrics
                st.divider()
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"Word Count: {word_count(report)}")
                with col2:
                    st.caption(f"Analysis Scope: {analysis_scope}")
                with col3:
                    st.caption(f"Generated: {st.session_state.analysis_date}")
            
            with tab2:
                st.markdown("### Source Context Used for Analysis")
                st.text_area(
                    "Wikipedia content used (first 500 chars per source):",
                    value="\n\n".join([f"[Source {i+1}]\n{p.content[:500]}..." 
                                      for i, p in enumerate(evaluated_pages)]),
                    height=400
                )
                
                st.markdown("### Source URLs")
                for i, page in enumerate(evaluated_pages):
                    st.write(f"{i+1}. {page.url}")
            
            # Download option
            st.download_button(
                label=" Download Report",
                data=report,
                file_name=f"industry_analysis_{industry.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
            
        except Exception as e:
            st.error(f"Report generation failed: {str(e)}")
            st.info("This might be due to API limits or content restrictions. Try a different industry or check your HF_TOKEN.")
            
            # Debug information
            with st.expander("Debug Information"):
                st.write("Error details:", str(e))
                st.write("Context length:", len(context))
                st.write("Number of sources:", len(evaluated_pages))

else:
    # Initial state - no industry provided
    st.info("Please enter an industry to begin analysis")
    
    # Example industries
    st.markdown("### Example Industries to Try:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.button("Renewable Energy", on_click=lambda: st.session_state.update({"example": "renewable energy"}))
        st.button("E-commerce", on_click=lambda: st.session_state.update({"example": "e-commerce"}))
    
    with col2:
        st.button("Artificial Intelligence", on_click=lambda: st.session_state.update({"example": "artificial intelligence"}))
        st.button("Pharmaceutical Industry", on_click=lambda: st.session_state.update({"example": "pharmaceutical industry"}))
    
    with col3:
        st.button("Fintech", on_click=lambda: st.session_state.update({"example": "fintech"}))
        st.button("Sustainable Agriculture", on_click=lambda: st.session_state.update({"example": "sustainable agriculture"}))
    
    # Handle example selection
    if "example" in st.session_state:
        st.experimental_set_query_params(industry=st.session_state.example)
        st.experimental_rerun()

# Footer
st.divider()
st.caption("""
**Professional Use Disclaimer**: This tool generates industry analysis based on Wikipedia content. 
Wikipedia may contain inaccuracies or outdated information. Always verify critical information 
through primary sources and expert consultation before making business decisions.
""")