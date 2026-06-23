import streamlit as st
import requests
import os
import sys
import json
import logging
from dotenv import load_dotenv

# Add parent workspace directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page configurations
st.set_page_config(
    page_title="PharmaAssist DSLM",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv(override=True)

logger = logging.getLogger("streamlit_app")

workspace_path = r"C:\Users\HP\projects\DSLM_Medical"

# Fallback direct importing of supervisor agent in case API is offline
@st.cache_resource
def get_local_supervisor():
    try:
        from agents.supervisor_agent import SupervisorAgent
        return SupervisorAgent(workspace_path=workspace_path)
    except Exception as e:
        logger.error(f"Failed to import/initialize local Supervisor Agent: {e}")
        return None

# --- Custom Premium CSS Injection ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
    }
    
    .main-header h1 {
        color: white !important;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .premium-card {
        background-color: var(--secondary-background-color, #f8fafc);
        color: var(--text-color, #0f172a);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .premium-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        border-color: #10b981;
    }
    
    .premium-card h3 {
        color: var(--text-color, #0f172a) !important;
        margin-top: 0;
    }
    
    .badge-essential {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-nonessential {
        background-color: #f1f5f9;
        color: #475569;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .source-card {
        border-left: 4px solid #10b981;
        background-color: var(--secondary-background-color, #ffffff);
        color: var(--text-color, #0f172a);
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border-top: 1px solid rgba(128, 128, 128, 0.1);
        border-right: 1px solid rgba(128, 128, 128, 0.1);
        border-bottom: 1px solid rgba(128, 128, 128, 0.1);
    }
    
    .source-snippet {
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: var(--text-color, #475569);
        opacity: 0.85;
    }
    
    .inner-snippet {
        margin-top: 1rem;
        padding: 0.75rem;
        background-color: var(--background-color, #ffffff);
        color: var(--text-color, #1e293b);
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.15);
    }
    
    .stChatInputContainer {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Backend Connectivity Solver ---
API_URL = "http://localhost:8000/api"
api_online = False

try:
    health_check = requests.get(f"{API_URL}/health", timeout=2)
    if health_check.status_code == 200:
        api_online = True
except Exception:
    api_online = False

# Sidebar Connection Info
st.sidebar.image("https://img.icons8.com/color/96/000000/pill.png", width=64)
st.sidebar.title("PharmaAssist DSLM")
st.sidebar.markdown("---")

if api_online:
    st.sidebar.success("🟢 API Server Online")
    mode = "api"
else:
    st.sidebar.warning("🟡 API Server Offline (Running Local Fallback Mode)")
    mode = "local"
    local_supervisor = get_local_supervisor()
    if local_supervisor is None:
        st.sidebar.error("🔴 Local Models failed to load! Ensure Ollama and Qdrant are configured.")

# Show Environment Configuration Details in Sidebar
st.sidebar.markdown("### Model Config")
st.sidebar.code(f"Model: {os.getenv('OLLAMA_MODEL', 'qwen2.5:1.5b-instruct-q4_K_M')}")
st.sidebar.code(f"Embeddings: BAAI/bge-small-en")
st.sidebar.code(f"Database: Qdrant Cloud Cluster")

# --- Application Header ---
st.markdown("""
<div class="main-header">
    <h1>PharmaAssist DSLM</h1>
    <p>Healthcare Domain-Specific Language Model and Intelligent Decision Support System</p>
</div>
""", unsafe_allow_html=True)

# Main Application Tabs
tab_chat, tab_interaction, tab_eml, tab_sources = st.tabs([
    "💬 Clinical Chat Assistant",
    "⚡ Drug Interaction Checker",
    "📋 WHO Essential Medicines EML",
    "🔍 Knowledge Payload Explorer"
])

# Utility function to run a query
def query_system(user_query: str) -> dict:
    if mode == "api":
        try:
            response = requests.post(f"{API_URL}/chat", json={"query": user_query})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"answer": f"API Error: {str(e)}", "sources": [], "query_type": "unknown", "success": False}
    else:
        if local_supervisor:
            res = local_supervisor.run(user_query)
            return {
                "answer": res["answer"],
                "sources": res["sources"],
                "query_type": res["query_type"],
                "success": True
            }
        else:
            return {"answer": "System is not initialized. Please ensure Ollama is running and start the app again.", "sources": [], "query_type": "unknown", "success": False}

# ================= TAB 1: Chat Assistant =================
with tab_chat:
    st.markdown("### Clinical Chat Interface")
    st.caption("Ask questions about indications, dosages, warnings, side effects, or drug classes.")
    
    # Session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Render chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Display sources if assistant message has them
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 View Citations & Sources"):
                    for idx, src in enumerate(msg["sources"]):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>[Doc {idx + 1}] {src['brand_name']} ({src['generic_name'].upper()})</strong><br>
                            <small>Section: {src['section']} | EML Category: {src['eml_section']}</small><br>
                            <p class="source-snippet">"{src['text_snippet']}"</p>
                        </div>
                        """, unsafe_allow_html=True)

    # Chat Input
    if user_input := st.chat_input("Enter drug name or question (e.g. What are the warnings for Amoxicillin?)..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        # Call RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("Analyzing pharmaceutical databases..."):
                res = query_system(user_input)
                st.markdown(res["answer"])
                
                # Show citations
                if res.get("sources"):
                    with st.expander("📚 View Citations & Sources"):
                        for idx, src in enumerate(res["sources"]):
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>[Doc {idx + 1}] {src['brand_name']} ({src['generic_name'].upper()})</strong><br>
                                <small>Section: {src['section']} | EML Category: {src['eml_section']}</small><br>
                                <p class="source-snippet">"{src['text_snippet']}"</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": res["answer"],
                "sources": res.get("sources", [])
            })

# ================= TAB 2: Interaction Checker =================
with tab_interaction:
    st.markdown("### Drug-to-Drug Interaction Checker")
    st.caption("Enter two or more drugs to analyze clinical interactions, contraindications, and warnings.")
    
    col1, col2 = st.columns(2)
    with col1:
        drug_a = st.text_input("First Drug Name (e.g. Amoxicillin)", key="drug_a")
    with col2:
        drug_b = st.text_input("Second Drug Name (e.g. Metformin)", key="drug_b")
        
    if st.button("Check Interactions ⚡", use_container_width=True):
        if not drug_a or not drug_b:
            st.error("Please enter both drug names to check interactions.")
        else:
            with st.spinner(f"Checking interactions between {drug_a} and {drug_b}..."):
                # Call specific interactions check query
                interaction_query = f"Can {drug_a} be taken with {drug_b}?"
                res = query_system(interaction_query)
                
                st.markdown("#### Clinical Analysis:")
                st.markdown(res["answer"])
                
                if res.get("sources"):
                    st.markdown("#### Sources Consulted:")
                    for idx, src in enumerate(res["sources"]):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>[Doc {idx + 1}] {src['brand_name']} ({src['generic_name'].upper()})</strong><br>
                            <small>Section: {src['section']} | ATC Codes: {src['atc_codes']}</small><br>
                            <p class="source-snippet">"{src['text_snippet']}"</p>
                        </div>
                        """, unsafe_allow_html=True)

# ================= TAB 3: WHO EML =================
with tab_eml:
    st.markdown("### WHO Essential Medicines EML lookup")
    st.caption("Search drugs in the WHO Essential Medicines database and discover alternatives within the same therapeutic class.")
    
    search_drug = st.text_input("Enter generic drug name (e.g. Amoxicillin)", key="eml_search_input")
    
    if st.button("Lookup Alternatives 🔍", use_container_width=True):
        if not search_drug:
            st.error("Please enter a generic name.")
        else:
            with st.spinner("Querying EML database..."):
                query = f"What is an alternative medicine for {search_drug}?"
                res = query_system(query)
                
                st.markdown(res["answer"])
                if res.get("sources"):
                    with st.expander("📚 View Supporting label details"):
                        for idx, src in enumerate(res["sources"]):
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>[Doc {idx + 1}] {src['brand_name']} ({src['generic_name'].upper()})</strong><br>
                                <small>Section: {src['section']}</small><br>
                                <p class="source-snippet">"{src['text_snippet']}"</p>
                            </div>
                            """, unsafe_allow_html=True)

# ================= TAB 4: Knowledge Payload Explorer =================
with tab_sources:
    st.markdown("### Knowledge Payload & Vector Store Explorer")
    st.caption("Perform semantic search to explore indexed OpenFDA label snippets directly from Qdrant Cloud.")
    
    keyword_query = st.text_input("Enter search phrase (e.g. diabetes, renal impairment, pediatric dosage)...")
    
    if st.button("Search Vector Payload 🌐"):
        if not keyword_query:
            st.error("Please enter a search phrase.")
        else:
            with st.spinner("Searching Qdrant Cloud..."):
                # Initialize pipeline components directly for exploration
                try:
                    from retrieval.retriever import HybridRetriever
                    retriever = HybridRetriever(workspace_path=workspace_path)
                    candidates = retriever.retrieve(keyword_query, limit=10)
                    
                    st.markdown(f"Found **{len(candidates)}** matching document chunks in Qdrant Cloud:")
                    
                    for idx, cand in enumerate(candidates):
                        meta = cand["metadata"]
                        is_essential = meta.get("is_essential", False)
                        badge_html = '<span class="badge-essential">Essential</span>' if is_essential else '<span class="badge-nonessential">Non-Essential</span>'
                        
                        st.markdown(f"""
                        <div class="premium-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <h3>{idx + 1}. {meta.get('brand_name', 'N/A')} ({meta.get('generic_name', 'N/A').upper()})</h3>
                                {badge_html}
                            </div>
                            <strong>Section:</strong> {meta.get('section', 'N/A').replace('_', ' ').title()}<br>
                            <strong>EML Section:</strong> {meta.get('eml_section', 'N/A')}<br>
                            <strong>ATC Code:</strong> {meta.get('atc_codes', 'N/A')}<br>
                            <strong>Qdrant Similarity Score:</strong> <code style="color:#059669;">{cand['score']:.4f}</code><br>
                            <p class="inner-snippet">
                                <em>"{cand['raw_content']}"</em>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error accessing database: {e}")
