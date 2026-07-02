import streamlit as st
from groq import Groq
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import create_client
import uuid

# ---- NEW: imports for live date/time + web search ----
from datetime import datetime
import streamlit.components.v1 as components
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
# ---- END NEW ----

# ---- PAGE CONFIG (must be first) ----
st.set_page_config(
    page_title="Greeny-AI ",
    page_icon="☘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- GLOBAL CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── ROOT VARIABLES ── */
:root {
    --bg-primary:    #0D0F0D;
    --bg-secondary:  #111311;
    --bg-card:       #1A1D1A;
    --bg-hover:      #1F231F;
    --accent:        #006241;
    --accent-light:  #00A86B;
    --accent-glow:   rgba(0, 98, 65, 0.25);
    --accent-subtle: rgba(0, 98, 65, 0.1);
    --text-primary:  #F0F2F0;
    --text-secondary:#9CA39C;
    --text-muted:    #5B605B;
    --border:        rgba(0, 98, 65, 0.25);
    --border-subtle: rgba(255,255,255,0.06);
    --success:       #00A86B;
    --radius:        14px;
    --radius-sm:     8px;
    --radius-pill:   999px;
}

/* ── FULL APP BACKGROUND ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

div[data-testid="stMain"] div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"] div.stButton button {
    background: #006241 !important;
    color: #ffffff !important;
    border-color: #006241 !important;
    padding: 0.25rem 0.5rem !important;
}

div[data-testid="stMain"] div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"] div.stButton button:hover {
    background: #004d33 !important;
    border-color: #004d33 !important;
    color: #ffffff !important;
}

div[data-testid="stMain"] div[data-testid="stHorizontalBlock"] {
    gap: 4px !important;
    padding: 0 !important;
}

div[data-testid="stMain"] div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"] {
    padding-left: 2px !important;
    padding-right: 2px !important;
    min-width: 0 !important;
    flex: 0 0 auto !important;
}                

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 70% 50% at 50% 0%,
        rgba(0, 98, 65, 0.08) 0%, transparent 65%),
        var(--bg-primary) !important;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1rem !important;
}

/* ── SIDEBAR HEADER BRAND ── */
.brand-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.3rem 0.4rem 0.6rem;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 0.6rem;
}
            
[data-testid="stSidebar"] hr {
    margin: 0.4rem 0 !important;
}

.brand-logo {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #1E3A34 0%, #000000 100%);
    border: 1px solid rgba(80, 220, 150, 0.25);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
   
    color: #00D084;
    box-shadow:
        0 0 20px rgba(0,168,107,.18),
        inset 0 1px rgba(255,255,255,.06);
}
.brand-name {
    font-size: 1.37rem;
    font-weight: 700;
    background: linear-gradient(90deg, #fff 0%, var(--accent-light) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.3px;
}

/* ── SIDEBAR SECTION LABELS ── */
.sidebar-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 0.5rem 0 0.3rem 0.2rem;
}

/* ── INPUTS & SELECTS ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextArea"] textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.4rem !important;
    transition: border-color 0.2s !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-light) !important;
}

/* ── CLEAR CONVERSATION BUTTON ── */
[data-testid="stSidebar"] .stButton > button {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 1.1rem !important;
    box-shadow: none !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--accent-subtle) !important;
    color: var(--accent-light) !important;
    border-color: var(--accent) !important;
}
            
/* First sidebar button */
[data-testid="stSidebar"] .stButton:first-of-type {
    margin-top: 12px;
}

/* ── DOWNLOAD BUTTON ── */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: all 0.2s !important;
}

[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--accent-light) !important;
    color: var(--accent-light) !important;
    background: var(--accent-subtle) !important;
}

/* ── MAIN AREA ── */
[data-testid="stMain"] .block-container {
    padding: 1.8rem 2.5rem !important;
    max-width: 860px !important;
}

/* ── TOP HEADER ── */
.top-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 1.5rem;
}

.header-left h1 {
    font-size: 1.55rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(90deg, #fff 30%, var(--accent-light) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    line-height: 1.2;
}

.header-left p {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin: 3px 0 0;
    font-weight: 400;
}

.model-pill {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 5px 14px;
    font-size: 0.75rem;
    color: var(--accent-light);
    font-weight: 600;
    letter-spacing: 0.3px;
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
}

.model-pill::before {
    content: '';
    width: 6px; height: 6px;
    background: var(--accent-light);
    border-radius: 50%;
    box-shadow: 0 0 6px var(--accent-light);
}

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.2rem 0 !important;
}

.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    background: transparent !important;
}

.stChatMessage:has([data-testid="chatAvatarIcon-user"]) > div:last-child > div {
    background: linear-gradient(135deg, var(--accent), #004d33) !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 0.75rem 1rem !important;
    color: #fff !important;
    box-shadow: 0 2px 12px var(--accent-glow) !important;
    max-width: 78% !important;
    margin-left: auto !important;
}

.stChatMessage:has([data-testid="chatAvatarIcon-assistant"]) > div:last-child > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 16px 16px 16px 4px !important;
    padding: 0.75rem 1rem !important;
    color: var(--text-primary) !important;
    max-width: 85% !important;
}

[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, var(--accent), var(--accent-light)) !important;
    border-radius: 10px !important;
}

[data-testid="chatAvatarIcon-assistant"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow), 0 4px 24px rgba(0,0,0,0.3) !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem !important;
    border: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
}

/* ── EMPTY STATE — centered like ChatGPT ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem 2rem 1.5rem;
    text-align: center;
    gap: 0.5rem;
}

.empty-title {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
}

section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
    background: #006241 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 999px !important;
    padding: 5px 10px !important;
    font-size: 0.75rem !important;
    white-space: nowrap !important;
    box-shadow: none !important;
}

[data-testid="stMain"] [data-testid="stHorizontalBlock"] {
    gap: 6px !important;
    justify-content: center !important;
}

[data-testid="stMain"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"] {
    padding: 0 !important;
    flex: 0 0 auto !important;
    width: auto !important;
}            

/* Override Streamlit column buttons to look like compact pills */
[data-testid="stMain"] [data-testid="stHorizontalBlock"] .stButton > button {
    background: #006241 !important;
    color: #fff !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-pill) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0rem 0rem !important;
    box-shadow: none !important;
    width: auto !important;
    transition: all 0.18s ease !important;
    white-space: nowrap !important;
}

[data-testid="stMain"] [data-testid="stHorizontalBlock"] .stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--text-primary) !important;
    background: var(--accent-subtle) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 10px var(--accent-glow) !important;
}

/* ── SUCCESS / INFO ALERTS ── */
[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
}

/* ── SPINNER ── */
[data-testid="stSpinner"] {
    color: var(--accent) !important;
}

/* ── DIVIDER ── */
hr {
    border-color: var(--border-subtle) !important;
    margin: 1rem 0 !important;
}

/* ── SCROLLBAR — always visible ── */
section[data-testid="stSidebar"] {
    scrollbar-color: #006241 #111311 !important;
    scrollbar-width: thin !important;
}

section[data-testid="stSidebar"]::-webkit-scrollbar {
    width: 5px !important;
    display: block !important;
}

section[data-testid="stSidebar"]::-webkit-scrollbar-track {
    background: #111311 !important;
}

section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
    background-color: #006241 !important;
    border-radius: 999px !important;
}

section[data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
    background-color: #00A86B !important;
}

@supports (scrollbar-color: auto auto) {
    section[data-testid="stSidebar"] {
        scrollbar-color: #006241 #111311 !important;
        scrollbar-width: thin !important;
    }
}

/* ── SELECTBOX DROPDOWN ── */
[data-baseweb="select"] {
    background: var(--bg-card) !important;
}

[data-baseweb="popover"] {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

[data-baseweb="menu"] {
    background: var(--bg-secondary) !important;
}

[role="option"]:hover {
    background: var(--bg-hover) !important;
}

/* ── SIDEBAR TEXT ── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
}
            
/* ── SIDEBAR CLOSE BUTTON ── */
/* ── SIDEBAR CLOSE BUTTON — always visible ── */
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
    opacity: 1 !important;
    visibility: visible !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button {
    opacity: 1 !important;
    visibility: visible !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.2s ease !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:hover {
    background: var(--accent-subtle) !important;
    border-color: var(--accent-light) !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button svg {
    opacity: 1 !important;
    fill: var(--accent-light) !important;
    stroke: var(--accent-light) !important;
}

[data-testid="collapsedControl"] {
    opacity: 1 !important;
    visibility: visible !important;
}

[data-testid="collapsedControl"] button {
    opacity: 1 !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

[data-testid="collapsedControl"] button svg {
    opacity: 1 !important;
    fill: var(--accent-light) !important;
    stroke: var(--accent-light) !important;
}
            
/* ── DOC BADGE ── */
.doc-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
    padding: 7px 10px;
    margin-bottom: 6px;
    font-size: 0.78rem;
    color: var(--text-secondary);
}

.doc-badge-icon { font-size: 14px; }

/* ── RAG INDICATOR ── */
.rag-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--accent-subtle);
    border: 1px solid rgba(0,98,65,0.4);
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 0.7rem;
    color: var(--accent-light);
    font-weight: 600;
    margin-bottom: 0.8rem;
}

</style>
""", unsafe_allow_html=True)

# ---- NEW: styling for the web-search toggle button is applied directly by
# the JS component further below (via st.iframe), not via CSS
# classes here — this avoids relying on Streamlit's container(key=) feature,
# which only exists in newer Streamlit versions. ----

# ---- SETUP ----
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()
supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

# ---- SUPABASE FUNCTIONS ----
def save_message(user_id, role, content):
    supabase.table("messages").insert({
        "user_id": user_id,
        "role": role,
        "content": content
    }).execute()

def load_messages(user_id):
    result = supabase.table("messages").select("role, content").eq(
        "user_id", user_id).order("id").execute()
    return [{"role": r["role"], "content": r["content"]} for r in result.data]

def clear_messages(user_id):
    supabase.table("messages").delete().eq("user_id", user_id).execute()

# ---- PDF UTILS ----
def load_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def search_chunks(query, chunks, embeddings, n=3):
    query_emb = embedder.encode([query])
    scores = np.dot(embeddings, query_emb.T).flatten()
    top_indices = scores.argsort()[-n:][::-1]
    return [chunks[i] for i in top_indices]

# ---- NEW: DATE/TIME + WEB SEARCH UTILS ----
def get_current_datetime_str():
    """Returns the real current date/time from the system clock."""
    return datetime.now().strftime("%A, %B %d, %Y - %I:%M %p")

def needs_web_search(query):
    """Lightweight keyword auto-detect for queries that likely need live info."""
    keywords = [
        "today", "current", "currently", "latest", "recent", "recently",
        "now", "this week", "this month", "this year", "right now",
        "news", "update", "weather", "score", "stock price", "exchange rate",
        "live", "breaking", "happening now", "as of today",
        "what date", "what's the date", "what time", "what's the time",
        "who is the current", "who is the president", "who is the prime minister"
    ]
    q = query.lower()
    return any(k in q for k in keywords)

def tavily_web_search(query, max_results=5):
    """
    Runs a live web search via Tavily and returns (context_text, sources_list).
    Fails silently (returns empty results) if the package or API key is missing,
    so the rest of the chat flow is never blocked.
    """
    if not TAVILY_AVAILABLE:
        return "", []
    if "TAVILY_API_KEY" not in st.secrets:
        return "", []
    try:
        client = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])
        response = client.search(query=query, max_results=max_results)
        results = response.get("results", [])
        context_parts = []
        sources = []
        for r in results:
            title = r.get("title", "Source")
            content = r.get("content", "")
            url = r.get("url", "")
            context_parts.append(f"Source: {title}\n{content}")
            sources.append({"title": title, "url": url})
        return "\n\n".join(context_parts), sources
    except Exception:
        return "", []
# ---- END NEW ----

# ---- SESSION STATE ----
if "pending_message" not in st.session_state:
    st.session_state.pending_message = ""

if "chips_used" not in st.session_state:
    st.session_state.chips_used = False

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    loaded = load_messages(st.session_state.user_id)
    st.session_state.messages = loaded

if "pdf_store" not in st.session_state:
    st.session_state.pdf_store = {}

# ---- NEW: web search toggle state ----
if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = False
# ---- END NEW ----

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("""
    <div class="brand-header">
        <div class="brand-logo">🍃</div>
        <div>
            <div class="brand-name">GreenyAI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑 Clear Conversation"):
        clear_messages(st.session_state.user_id)
        st.session_state.messages = []
        st.session_state.chips_used = False
        st.rerun()

    st.markdown('<div class="sidebar-label">Model</div>', unsafe_allow_html=True)
    model_options = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile"
    ]
    selected_model = st.selectbox("", model_options, label_visibility="collapsed")

    st.markdown('<div class="sidebar-label">System Prompt</div>', unsafe_allow_html=True)
    system_prompt = st.text_area(
        "", value="You are a helpful assistant.",
        height=68, label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDFs here", type=["pdf"],
        accept_multiple_files=True, label_visibility="collapsed"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.pdf_store:
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    text = load_pdf(uploaded_file)
                    chunks = chunk_text(text)
                    embeddings = embedder.encode(chunks)
                    st.session_state.pdf_store[uploaded_file.name] = {
                        "chunks": chunks,
                        "embeddings": embeddings
                    }
                st.success(f"✓ {uploaded_file.name}")

    selected_pdf = None
    if st.session_state.pdf_store:
        st.markdown('<div class="sidebar-label">Active Document</div>', unsafe_allow_html=True)
        for doc_name in st.session_state.pdf_store.keys():
            st.markdown(f"""
            <div class="doc-badge">
                <span class="doc-badge-icon">📄</span>
                {doc_name[:28]}{'...' if len(doc_name) > 28 else ''}
            </div>
            """, unsafe_allow_html=True)
        selected_pdf = st.selectbox(
            "Search in:", list(st.session_state.pdf_store.keys()),
            label_visibility="collapsed"
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    if st.session_state.messages:
        st.markdown('<div class="sidebar-label">Export</div>', unsafe_allow_html=True)
        chat_text = "\n\n".join([
            f"{'You' if m['role'] == 'user' else 'GreenyAI'}:\n{m['content']}"
            for m in st.session_state.messages
        ])
        st.download_button(
            "⬇ Download Chat", data=chat_text,
            file_name="Greenyai_chat.txt", mime="text/plain"
        )

    st.markdown("<hr>", unsafe_allow_html=True)

# ---- MAIN AREA ----
st.markdown(f"""
<div class="top-header">
    <div class="header-left">
         <h1 style="background:none;-webkit-text-fill-color:unset;">🍃 <span style="color:#00A86B;">Greeny</span><span style="color:#fff;">AI</span> <span style="color:#fff;">Chat</span></h1>
        <p>Ask anything · Upload PDFs · Switch models on the fly</p>
    </div>
    <div class="model-pill">{selected_model}</div>
</div>
""", unsafe_allow_html=True)

if selected_pdf:
    st.markdown(f"""
    <div class="rag-pill">⚡ RAG active — querying <strong>{selected_pdf[:30]}</strong></div>
    """, unsafe_allow_html=True)

if st.session_state.web_search_enabled:
    st.markdown("""
    <div class="rag-pill">🌐 Live web search — ON</div>
    """, unsafe_allow_html=True)

# ---- STEP 1: CAPTURE INPUT FIRST ----
pending = st.session_state.get("pending_message", "")
if pending:
    st.session_state.pending_message = ""
    st.session_state.chips_used = True

# ---- NEW: WEB SEARCH TOGGLE BUTTON (placed left of the chat bar) ----
# Plain st.button (no container/key-class dependency) — works on every Streamlit version.
if st.button("🌐", key="web_toggle_btn", help="Toggle live web search"):
    st.session_state.web_search_enabled = not st.session_state.web_search_enabled
    st.rerun()

# components.v1.html is the correct/supported way to run custom JS in Streamlit
# (unlike scripts injected via st.markdown, which browsers often silently ignore).
# It finds the button by its emoji text — no reliance on any Streamlit-version-
# specific CSS class — styles it, then pins it beside the real chat input.
_web_toggle_on_js = "true" if st.session_state.web_search_enabled else "false"

components.html(
    """
    <script>
    (function() {
        const enabled = """ + _web_toggle_on_js + """;

        function styleAndPositionWebToggle() {
            const parentDoc = window.parent.document;
            const buttons = parentDoc.querySelectorAll('button');
            let toggleBtn = null;
            buttons.forEach(function(b) {
                if (b.textContent.trim() === '\U0001F310') { toggleBtn = b; }
            });
            if (!toggleBtn) return;

            toggleBtn.style.setProperty('width', '44px', 'important');
            toggleBtn.style.setProperty('height', '44px', 'important');
            toggleBtn.style.setProperty('min-height', '44px', 'important');
            toggleBtn.style.setProperty('border-radius', '50%', 'important');
            toggleBtn.style.setProperty('font-size', '1.15rem', 'important');
            toggleBtn.style.setProperty('padding', '0', 'important');
            toggleBtn.style.setProperty('display', 'flex', 'important');
            toggleBtn.style.setProperty('align-items', 'center', 'important');
            toggleBtn.style.setProperty('justify-content', 'center', 'important');

            if (enabled) {
                toggleBtn.style.setProperty('background', 'linear-gradient(135deg, #006241, #00A86B)', 'important');
                toggleBtn.style.setProperty('border', '1px solid #00A86B', 'important');
                toggleBtn.style.setProperty('color', '#fff', 'important');
                toggleBtn.style.setProperty('box-shadow', '0 0 14px rgba(0,98,65,0.4)', 'important');
            } else {
                toggleBtn.style.setProperty('background', '#1A1D1A', 'important');
                toggleBtn.style.setProperty('border', '1px solid rgba(0,98,65,0.25)', 'important');
                toggleBtn.style.setProperty('color', '#9CA39C', 'important');
                toggleBtn.style.setProperty('box-shadow', 'none', 'important');
            }

            const wrapper = toggleBtn.closest('div[data-testid="stElementContainer"], div.element-container') || toggleBtn;
            wrapper.style.setProperty('position', 'fixed', 'important');
            wrapper.style.setProperty('width', 'auto', 'important');
            wrapper.style.setProperty('z-index', '1000', 'important');

            const chatInput = parentDoc.querySelector('[data-testid="stChatInput"]');
            if (chatInput) {
                const rect = chatInput.getBoundingClientRect();
                wrapper.style.setProperty('top', (rect.top + rect.height / 2 - 22) + 'px', 'important');
                wrapper.style.setProperty('left', (rect.left - 54) + 'px', 'important');
                wrapper.style.setProperty('bottom', 'auto', 'important');
            } else {
                wrapper.style.setProperty('bottom', '18px', 'important');
                wrapper.style.setProperty('left', '20px', 'important');
                wrapper.style.setProperty('top', 'auto', 'important');
            }
        }

        styleAndPositionWebToggle();
        setInterval(styleAndPositionWebToggle, 400);
        window.parent.addEventListener('resize', styleAndPositionWebToggle);
    })();
    </script>
    """,
    height=0,
)

user_input = st.chat_input("Ask anything...")

if user_input:
    st.session_state.chips_used = True

# REPLACE your chips block with this:
if not st.session_state.chips_used:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-title">Where should we begin?</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1:
            if st.button("📄 Summarize"):
                st.session_state.pending_message = "Please summarize the uploaded document for me."
                st.session_state.chips_used = True
                st.rerun()
        with c2:
            if st.button("💻 Write code"):
                st.session_state.pending_message = "Write me a Python function that"
                st.session_state.chips_used = True
                st.rerun()
        with c3:
            if st.button("🏖️ Best Places"):
                st.session_state.pending_message = "What are the best places to visit in this area?"
                st.session_state.chips_used = True
                st.rerun()
        with c4:
            if st.button("🔍 Compare"):
                st.session_state.pending_message = "Compare and contrast these two ideas:"
                st.session_state.chips_used = True
                st.rerun()

st.markdown("""
<script>
function styleChips() {
    const buttons = window.parent.document.querySelectorAll(
        '[data-testid="stMain"] [data-testid="stHorizontalBlock"] button'
    );
    buttons.forEach(btn => {
        btn.style.setProperty('background-color', '#006241', 'important');
        btn.style.setProperty('color', '#ffffff', 'important');
        btn.style.setProperty('border', 'none', 'important');
        btn.style.setProperty('border-radius', '999px', 'important');
        btn.style.setProperty('padding', '5px 10px', 'important');
        btn.style.setProperty('font-size', '0.75rem', 'important');
        btn.style.setProperty('box-shadow', 'none', 'important');
    });
}
styleChips();
setTimeout(styleChips, 300);
setTimeout(styleChips, 800);
</script>
""", unsafe_allow_html=True)

# ---- STEP 3: DISPLAY MESSAGES ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---- STEP 4: PROCESS MESSAGE ----
if not user_input and not pending:
    st.stop()

if not user_input and pending:
    user_input = pending

st.session_state.messages.append({"role": "user", "content": user_input})
save_message(st.session_state.user_id, "user", user_input)

with st.chat_message("user"):
    st.markdown(user_input)

with st.chat_message("assistant"):
    placeholder = st.empty()
    full_response = ""
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    effective_system_prompt = system_prompt + (
        f"\n\nCurrent date and time: {get_current_datetime_str()}. "
        "Always use this exact date/time if the user asks what today's date, "
        "the day of the week, or the current time is — never guess or rely on "
        "your training cutoff for this."
    )

  
    use_web_search = st.session_state.web_search_enabled or needs_web_search(user_input)
    web_context, web_sources = "", []
    if use_web_search:
        with st.spinner("Searching the web..."):
            web_context, web_sources = tavily_web_search(user_input)


    if selected_pdf and selected_pdf in st.session_state.pdf_store:
        store = st.session_state.pdf_store[selected_pdf]
        relevant = search_chunks(user_input, store["chunks"], store["embeddings"])
        context = "\n\n".join(relevant)
       
        if web_context:
            context += f"\n\nLive web results:\n{web_context}"
        
        rag_prompt = f"Use this context to answer:\n\n{context}\n\nQuestion: {user_input}"
        messages_to_send = (
            [{"role": "system", "content": effective_system_prompt}]
            + st.session_state.messages[:-1]
            + [{"role": "user", "content": rag_prompt}]
        )
    elif web_context:
        web_prompt = f"Use these live web search results to answer accurately:\n\n{web_context}\n\nQuestion: {user_input}"
        messages_to_send = (
            [{"role": "system", "content": effective_system_prompt}]
            + st.session_state.messages[:-1]
            + [{"role": "user", "content": web_prompt}]
        )
 
    else:
        messages_to_send = (
            [{"role": "system", "content": effective_system_prompt}]
            + st.session_state.messages
        )

    stream = groq_client.chat.completions.create(
        model=selected_model,
        messages=messages_to_send,
        stream=True
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
            placeholder.markdown(full_response + "▌")
    placeholder.markdown(full_response)

    if web_sources:
        sources_md = "\n\n---\n**🌐 Sources:**\n" + "\n".join(
            [f"- [{s['title']}]({s['url']})" for s in web_sources if s.get("url")]
        )
        full_response += sources_md
        placeholder.markdown(full_response)
  
st.session_state.messages.append({"role": "assistant", "content": full_response})
save_message(st.session_state.user_id, "assistant", full_response)
