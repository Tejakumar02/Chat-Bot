import streamlit as st
from groq import Groq
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import create_client
import uuid
import time

# ---- NEW: imports for live date/time + web search ----
from datetime import datetime
import streamlit.components.v1 as components
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
# ---- END NEW ----

# ---- NEW: import for Gemini vision (image understanding) ----
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
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
    padding: 2rem 2.5rem !important;
    max-width: 860px !important;
}

/* NEW: Streamlit reserves a tall default header bar above .block-container.
Hiding stToolbar's icons first, then collapsing the now-empty header down to
0 height, avoids the old glitch of clipped hamburger/Deploy icon fragments.
Safe to fully collapse (not just shrink to 2rem): the sidebar's own collapse
button lives inside stSidebar (styled separately below) and the floating
reopen control is its own independent element — neither sits inside, or
depends on the height of, stHeader. */
[data-testid="stToolbar"] {
    display: block !important;
}
[data-testid="stHeader"] {
    height: auto !important;
    min-height: auto !important;
}

/* NEW: Streamlit adds a default top gap before the first block in the main
area (independent of stHeader). Zeroing it out on the very first element
container in stMain is what actually pulls the model pill up to the true
top of the page, rather than leaving a residual gap even after stHeader is
collapsed above. */
[data-testid="stMain"] .block-container > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* NEW: raw <iframe> elements get a visible default border from the browser's
own stylesheet unless explicitly reset — this app has 3 zero-content utility
iframes (for icon positioning + button tagging), each just 1px tall, so an
unreset default border on each would show as exactly the kind of small line
fragments being reported. */
iframe {
    border: none !important;
    display: block !important;
}

/* NEW (superseded twice): overflow/scrollbar-hiding CSS on stIFrame was
tried first and didn't work — a native iframe's scrollbar is controlled by
its `scrolling` HTML attribute, which outer CSS can't reliably override.
Switching to components.html(scrolling=False) was tried next, but that API
is deprecated and removed after 2026-06-01 in favor of st.iframe(). The
actual, lasting fix: st.iframe() hardcodes scrolling=True for raw HTML with
no override (confirmed in Streamlit's source), so instead each srcdoc string
below now opens with a small inline CSS reset (html/body margin, padding,
and height zeroed, overflow hidden) — with the default body margin zeroed
out, there's nothing left to overflow, so the scrollbar never appears
regardless of the scrolling attribute. This CSS block is left in as a
harmless no-op belt-and-braces rule, not as the fix itself.
NOTE: never spell out a literal closing style-tag sequence inside this
comment (or anywhere else in this global stylesheet) — the browser's HTML
parser treats <style> content as raw text and closes the tag on the first
literal match, even inside a /* comment */. Doing that once is exactly what
dumped all the CSS below as visible page text. */
[data-testid="stIFrame"] {
    overflow: hidden !important;
    scrollbar-width: none !important;
}
[data-testid="stIFrame"]::-webkit-scrollbar {
    width: 0 !important;
    height: 0 !important;
    display: none !important;
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
    padding: 0rem 2rem 0.3rem;
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
# the JS component further below (via st.components.v1.html), not via CSS
# classes here — this avoids relying on Streamlit's container(key=) feature,
# which only exists in newer Streamlit versions. ----

# ---- SETUP ----
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()
supabase = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

# ---- NEW: admin password, sourced from secrets like every other credential in
# this app (never hardcode a password in source). Add to .streamlit/secrets.toml:
#   ADMIN_PASSWORD = "your-password-here"
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD")
# ---- END NEW ----

# ---- NEW: chat-session-aware Supabase functions ----
# Requires a `chat_id` (text) column on your `messages` table. In the Supabase
# SQL editor, run:  ALTER TABLE messages ADD COLUMN chat_id text;
# Existing rows saved before this column existed will have chat_id = NULL —
# those get grouped into a "Previous Chat" entry on load so nothing is lost.
def save_message(user_id, chat_id, role, content):
    supabase.table("messages").insert({
        "user_id": user_id,
        "chat_id": chat_id,
        "role": role,
        "content": content
    }).execute()

def _derive_chat_title(messages):
    """Auto-generates a chat title from the first user message, ChatGPT-style."""
    for m in messages:
        if m["role"] == "user":
            text = m["content"].strip().replace("\n", " ")
            return text[:40] + ("…" if len(text) > 40 else "")
    return "New Chat"

def load_chats(user_id):
    """
    Loads every saved message for this user and groups it into chats by
    chat_id. Rows with no chat_id (saved before this feature existed) are
    grouped under a special "legacy" key so old history isn't lost.
    """
    result = supabase.table("messages").select("id, chat_id, role, content").eq(
        "user_id", user_id).order("id").execute()
    chats = {}
    for row in result.data:
        cid = row["chat_id"] or "legacy"
        if cid not in chats:
            # NEW: pdf_store/image_store live per-chat now (not persisted to
            # Supabase — uploads were always in-memory only), so every chat,
            # including ones restored from history, starts with its own
            # empty file stores rather than sharing one global store.
            chats[cid] = {"messages": [], "created": row["id"], "pdf_store": {}, "image_store": {}}
        chats[cid]["messages"].append({"role": row["role"], "content": row["content"]})
    for cid, chat in chats.items():
        chat["title"] = "Previous Chat" if cid == "legacy" else _derive_chat_title(chat["messages"])
    return chats

def delete_chat_messages(user_id, chat_id):
    query = supabase.table("messages").delete().eq("user_id", user_id)
    if chat_id == "legacy":
        # "legacy" is just a Python-side grouping label for old rows saved
        # before chat_id existed — those rows actually have chat_id = NULL
        # in the database, not the string "legacy".
        query = query.is_("chat_id", "null")
    else:
        query = query.eq("chat_id", chat_id)
    query.execute()
# ---- END NEW ----

# ---- NEW: chat session management ----
def new_chat():
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = {
        "title": "New Chat",
        "created": time.time(),
        "messages": [],
        # NEW: each chat owns its own file stores, so a new chat always
        # starts with nothing attached — no leftover PDFs/images from
        # whatever chat you were just in.
        "pdf_store": {},
        "image_store": {},
    }
    st.session_state.current_chat_id = new_id
    st.session_state.chips_used = False  # show the empty-state welcome screen again
    # NEW: clear the sidebar file selectors too, since they're widget-level
    # session state that would otherwise still show the old chat's selection
    # until the user manually changes it.
    st.session_state["pdf_selectbox"] = NONE_OPTION
    st.session_state["image_selectbox"] = NONE_OPTION

def delete_chat(chat_id):
    delete_chat_messages(st.session_state.user_id, chat_id)
    if chat_id in st.session_state.chats:
        # NOTE: pdf_store/image_store live inside this chat's dict now, so
        # deleting the chat entry automatically drops its uploaded files too
        # — no separate cleanup step needed. Files were in-memory only
        # (never written to Supabase), so nothing needs deleting there either.
        del st.session_state.chats[chat_id]
    # Deleting the chat you're currently viewing auto-creates a fresh one.
    if st.session_state.current_chat_id == chat_id or not st.session_state.chats:
        new_chat()
# ---- END NEW ----

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

def gemini_vision_answer(user_input, image_bytes, mime_type, system_instruction, placeholder,
                          extra_context="", temperature=0.7, top_p=1.0, max_output_tokens=1024):
    """
    Streams an answer from Gemini (gemini-2.5-flash) using the uploaded image,
    the user's question, and any RAG/web context, writing tokens into the given
    Streamlit placeholder as they arrive. Returns the full response text.

    Gemini handles image + text + extra context natively in a single call, so
    this one function covers every "image present" combination from the routing
    spec (image alone, image+PDF, image+web, image+PDF+web) without needing a
    separate caption-then-reason hop.

    temperature/top_p/max_output_tokens mirror the same admin-configured
    generation settings used for the Groq/text path, so switching between
    RAG and Vision doesn't silently change how "creative" responses are.
    """
    if not GEMINI_AVAILABLE:
        msg = "⚠️ Vision support isn't installed. Add `google-genai` to requirements.txt."
        placeholder.markdown(msg)
        return msg
    if "GEMINI_API_KEY" not in st.secrets:
        msg = "⚠️ Vision support needs a `GEMINI_API_KEY` in secrets.toml."
        placeholder.markdown(msg)
        return msg
    try:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        prompt_text = user_input
        if extra_context:
            prompt_text = f"{extra_context}\n\nQuestion: {user_input}"

        full_response = ""
        stream = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=[
                prompt_text,
                genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config=genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
            ),
        )
        for chunk in stream:
            if chunk.text:
                full_response += chunk.text
                placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)
        return full_response
    except Exception as e:
        error_msg = f"⚠️ Vision request failed: {e}"
        placeholder.markdown(error_msg)
        return error_msg
# ---- END NEW ----

# ---- NEW: sentinel used to let PDF / Image selectboxes be explicitly turned off ----
NONE_OPTION = "— None (off) —"
# ---- END NEW ----

# ---- NEW: strict either/or — picking one mode auto-deactivates the other ----
def _on_pdf_selectbox_change():
    if st.session_state.get("pdf_selectbox") not in (None, NONE_OPTION):
        st.session_state["image_selectbox"] = NONE_OPTION

def _on_image_selectbox_change():
    if st.session_state.get("image_selectbox") not in (None, NONE_OPTION):
        st.session_state["pdf_selectbox"] = NONE_OPTION
# ---- END NEW ----

# ---- NEW: unified attachment handler for the chat_input's native file
# upload. Given one UploadedFile coming back from st.chat_input(accept_file=
# ...), routes it into the CURRENT chat's pdf_store or image_store by
# extension, reusing the exact same load_pdf/chunk_text/embedder pipeline
# the old plus-menu uploaders used, then marks it as the active selection
# (mirroring the old strict either/or behavior between the two selectboxes). ----
def _handle_attached_file(uploaded_file):
    chat = st.session_state.chats[st.session_state.current_chat_id]
    name = uploaded_file.name
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    if ext == "pdf":
        if name not in chat["pdf_store"]:
            text = load_pdf(uploaded_file)
            chunks = chunk_text(text)
            embeddings = embedder.encode(chunks)
            chat["pdf_store"][name] = {"chunks": chunks, "embeddings": embeddings}
        st.session_state["pdf_selectbox"] = name
        st.session_state["image_selectbox"] = NONE_OPTION
    else:
        if name not in chat["image_store"]:
            chat["image_store"][name] = {
                "bytes": uploaded_file.getvalue(),
                "mime_type": uploaded_file.type or f"image/{ext}",
            }
        st.session_state["image_selectbox"] = name
        st.session_state["pdf_selectbox"] = NONE_OPTION
# ---- END NEW ----

# ---- SESSION STATE ----
if "pending_message" not in st.session_state:
    st.session_state.pending_message = ""

if "chips_used" not in st.session_state:
    st.session_state.chips_used = False

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# ---- NEW: multi-chat session state ----
if "chats" not in st.session_state:
    st.session_state.chats = load_chats(st.session_state.user_id)

if "current_chat_id" not in st.session_state:
    if st.session_state.chats:
        # Most recently created chat becomes active on load.
        st.session_state.current_chat_id = max(
            st.session_state.chats.items(), key=lambda kv: kv[1]["created"]
        )[0]
        # Correct the naive chips_used=False default for returning users whose
        # most recent chat already has history — show messages, not the
        # empty-state welcome screen.
        active = st.session_state.chats[st.session_state.current_chat_id]
        st.session_state.chips_used = len(active["messages"]) > 0
    else:
        new_chat()
# ---- END NEW ----

# NOTE: pdf_store/image_store used to be initialized here as global,
# shared-across-all-chats dicts. They now live inside each chat's own dict
# (set in new_chat()/load_chats()), so there's nothing global to initialize
# — every chat, old or new, already carries its own "pdf_store"/"image_store".

# ---- NEW: web search toggle state ----
if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = False
# ---- END NEW ----

# ---- NEW: admin-gated settings — these must be initialized here (not just
# inside the admin panel) since STEP 4 reads them every run regardless of
# whether the admin panel has ever been unlocked. ----
if "admin_unlocked" not in st.session_state:
    st.session_state.admin_unlocked = False
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful assistant."
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "top_p" not in st.session_state:
    st.session_state.top_p = 1.0
if "num_predict" not in st.session_state:
    st.session_state.num_predict = 1024
# ---- END NEW ----

# ---- NEW: model selection moved to the sidebar (right after the logo) —
# state initialized here since the sidebar selectbox needs it immediately. ----
model_options = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile"
]
if "selected_model" not in st.session_state:
    st.session_state.selected_model = model_options[0]
# ---- END NEW ----

# ---- NEW: capture the chat input (and any attached file) BEFORE the sidebar
# runs. This ordering is required, not cosmetic:
#   1) The sidebar instantiates the `pdf_selectbox`/`image_selectbox` widgets
#      further down. Streamlit forbids writing to a widget's session_state
#      key AFTER that widget has been instantiated in the same run — doing
#      so raises StreamlitAPIException. Handling the attachment here, before
#      the sidebar runs, means the write always happens pre-instantiation.
#   2) It also means `selected_image`/`selected_pdf` (computed in the sidebar
#      from these same keys) reflect what was JUST attached in this exact
#      turn — so the VLM/RAG routing below reacts to this message's
#      attachment immediately, instead of lagging one turn behind.
# st.chat_input() always renders pinned to the bottom of the page regardless
# of where in the script it's called, so moving the call up here does not
# change anything visually — the widget still appears in the same place.
chat_value = st.chat_input(
    "Ask anything...",
    accept_file=True,
    file_type=["pdf", "png", "jpg", "jpeg", "webp"],
)

user_input = ""
if chat_value:
    for f in chat_value.files:
        _handle_attached_file(f)
    user_input = chat_value.text or ""
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

    # ---- NEW: Model selection — moved here from the main bar ----
    st.markdown('<div class="sidebar-label">Model</div>', unsafe_allow_html=True)
    st.session_state.selected_model = st.selectbox(
        "", model_options,
        index=model_options.index(st.session_state.selected_model),
        label_visibility="collapsed", key="sidebar_model_select"
    )
    # ---- END NEW ----

    # ---- NEW: New Chat button + chat session list (replaces Clear Conversation) ----
    if st.button("✙  New chat", use_container_width=True, key="new_chat_btn"):
        new_chat()
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Your Chats")

    chat_items = sorted(
        st.session_state.chats.items(), key=lambda kv: kv[1]["created"], reverse=True
    )
    for chat_id, chat in chat_items:
        is_active = chat_id == st.session_state.current_chat_id
        label = ("● " if is_active else "") + chat["title"]
        col_select, col_delete = st.columns([3.5, 1])
        with col_select:
            if st.button(label, key=f"select_{chat_id}", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.session_state.chips_used = len(chat["messages"]) > 0
                # NEW: each chat has its own pdf_store/image_store now, so
                # reset the selector widgets rather than carrying over
                # whatever was selected in the chat you're leaving.
                st.session_state["pdf_selectbox"] = NONE_OPTION
                st.session_state["image_selectbox"] = NONE_OPTION
                st.rerun()
        with col_delete:
            if st.button(" 🗑️ ", key=f"del_{chat_id}"):
                delete_chat(chat_id)
                st.rerun()
    # ---- END NEW ----

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---- NEW: Admin Settings (password-gated) ----
    # System prompt / temperature / top_p / num_predict are read from
    # st.session_state everywhere downstream, so they still have valid values
    # even when this panel is locked and none of these widgets are rendering.
    with st.expander("⚙️ Admin settings"):
        if not st.session_state.admin_unlocked:
            st.caption("Locked. Enter the admin password to change these.")
            pw = st.text_input("Password", type="password", key="admin_pw_input")
            if st.button("Unlock", key="admin_unlock_btn"):
                if ADMIN_PASSWORD is not None and pw == ADMIN_PASSWORD:
                    st.session_state.admin_unlocked = True
                    st.rerun()
                elif ADMIN_PASSWORD is None:
                    st.error("No ADMIN_PASSWORD set in secrets.toml — admin panel can't unlock.")
                else:
                    st.error("Incorrect password.")
        else:
            st.markdown('<div class="sidebar-label">System Prompt</div>', unsafe_allow_html=True)
            st.session_state.system_prompt = st.text_area(
                "", value=st.session_state.system_prompt,
                height=68, label_visibility="collapsed"
            )

            st.session_state.temperature = st.slider(
                "Temperature", 0.0, 2.0, st.session_state.temperature, 0.05,
                help="Higher = more random/creative. Lower = more focused/deterministic.",
            )
            st.session_state.top_p = st.slider(
                "Top-p", 0.05, 1.0, st.session_state.top_p, 0.05,
            )
            st.session_state.num_predict = st.slider(
                "Max response tokens", 128, 4096, st.session_state.num_predict, 128,
            )

            # NOTE: this app supports multiple PDFs (pdf_store), unlike a
            # single-document model — this clears only the currently ACTIVE
            # one, not every uploaded PDF. Tell me if you actually want a
            # bulk "clear everything" button instead.
            active_pdf_name = st.session_state.get("pdf_selectbox")
            if active_pdf_name and active_pdf_name != NONE_OPTION and st.button("Clear active document"):
                del st.session_state.chats[st.session_state.current_chat_id]["pdf_store"][active_pdf_name]
                st.session_state["pdf_selectbox"] = NONE_OPTION
                st.rerun()

            if st.button("🔒 Lock settings"):
                st.session_state.admin_unlocked = False
                st.rerun()
    # ---- END NEW ----

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---- NEW: file uploaders removed from the sidebar — uploading now happens
    # exclusively through the "+" attach icon built into the chat input bar
    # itself. The "Active" selectboxes below read from the CURRENT CHAT's own
    # pdf_store/image_store (not a global one), so switching chats only ever
    # shows that chat's own files, and deleting a chat takes its files with it. ----
    current_pdf_store = st.session_state.chats[st.session_state.current_chat_id]["pdf_store"]
    current_image_store = st.session_state.chats[st.session_state.current_chat_id]["image_store"]

    selected_pdf = None
    if current_pdf_store:
        st.markdown('<div class="sidebar-label">Active Document</div>', unsafe_allow_html=True)
        for doc_name in current_pdf_store.keys():
            st.markdown(f"""
            <div class="doc-badge">
                <span class="doc-badge-icon">📄</span>
                {doc_name[:28]}{'...' if len(doc_name) > 28 else ''}
            </div>
            """, unsafe_allow_html=True)
        pdf_options = [NONE_OPTION] + list(current_pdf_store.keys())
        pdf_choice = st.selectbox(
            "Search in:", pdf_options,
            label_visibility="collapsed", key="pdf_selectbox",
            on_change=_on_pdf_selectbox_change
        )
        selected_pdf = None if pdf_choice == NONE_OPTION else pdf_choice

    selected_image = None
    if current_image_store:
        st.markdown('<div class="sidebar-label">Active Image</div>', unsafe_allow_html=True)
        for img_name in current_image_store.keys():
            st.markdown(f"""
            <div class="doc-badge">
                <span class="doc-badge-icon">🖼️</span>
                {img_name[:28]}{'...' if len(img_name) > 28 else ''}
            </div>
            """, unsafe_allow_html=True)
        image_options = [NONE_OPTION] + list(current_image_store.keys())
        image_choice = st.selectbox(
            "Analyze:", image_options,
            label_visibility="collapsed", key="image_selectbox",
            on_change=_on_image_selectbox_change
        )
        selected_image = None if image_choice == NONE_OPTION else image_choice
    # ---- END NEW ----

    st.markdown("<hr>", unsafe_allow_html=True)

    current_messages = st.session_state.chats[st.session_state.current_chat_id]["messages"]
    if current_messages:
        st.markdown('<div class="sidebar-label">Export</div>', unsafe_allow_html=True)
        chat_text = "\n\n".join([
            f"{'You' if m['role'] == 'user' else 'GreenyAI'}:\n{m['content']}"
            for m in current_messages
        ])
        st.download_button(
            "⬇ Download Chat", data=chat_text,
            file_name="Greenyai_chat.txt", mime="text/plain"
        )

    st.markdown("<hr>", unsafe_allow_html=True)

# ---- NEW: chat-delete (🗑️) button styling ----
# Each delete button has a per-chat dynamic key (del_<chat_id>), so there's no
# single stable st-key-* class to target directly. Instead, a small JS pass
# tags every button containing the trash icon with a shared class, then plain
# CSS (the reliable approach, not JS inline styles) handles the actual look.
st.markdown("""
<style>
[data-testid="stSidebar"] button.greeny-chat-delete-btn,
[data-testid="stSidebar"] button.greeny-chat-delete-btn:hover,
[data-testid="stSidebar"] button.greeny-chat-delete-btn:focus,
[data-testid="stSidebar"] button.greeny-chat-delete-btn:active {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    width: auto !important;
}
[data-testid="stSidebar"] button.greeny-chat-delete-btn,
[data-testid="stSidebar"] button.greeny-chat-delete-btn * {
    font-size: 1.3rem !important;
    line-height: 1 !important;
}
[data-testid="stSidebar"] button.greeny-chat-delete-btn:hover,
[data-testid="stSidebar"] button.greeny-chat-delete-btn:hover * {
    color: #e05252 !important;
    cursor: pointer;
}
[data-testid="stSidebar"] button.greeny-chat-delete-btn::after {
    content: " Delete chat";
    font-size: 0.75rem !important;
    font-weight: 400 !important;
    color: transparent !important;
    transition: color 0.15s ease !important;
}
[data-testid="stSidebar"] button.greeny-chat-delete-btn:hover::after {
    color: #9CA39C !important;
}
</style>
""", unsafe_allow_html=True)

st.iframe(
    """
    <style>html, body { margin: 0; padding: 0; height: 0; overflow: hidden; }</style>
    <script>
    (function() {
        function tagDeleteButtons() {
            const parentDoc = window.parent.document;
            const buttons = parentDoc.querySelectorAll('button');
            buttons.forEach(function(b) {
                if (b.textContent.indexOf('\U0001F5D1') !== -1) {
                    b.classList.add('greeny-chat-delete-btn');
                }
            });
        }
        tagDeleteButtons();
        setInterval(tagDeleteButtons, 400);
    })();
    </script>
    """,
    height=1,
)

selected_model = st.session_state.selected_model

st.markdown(f"""
<div class="top-header" style="border-bottom:none; margin-top:0; padding-top:0; margin-bottom:0.3rem; padding-bottom:0.3rem; justify-content:flex-end;">
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

# ----  vision-active status pill ----
if selected_image:
    st.markdown(f"""
    <div class="rag-pill">🖼️ Vision active — analyzing <strong>{selected_image[:30]}</strong></div>
    """, unsafe_allow_html=True)

# ---- STEP 1: CAPTURE INPUT FIRST ----
pending = st.session_state.get("pending_message", "")
if pending:
    st.session_state.pending_message = ""
    st.session_state.chips_used = True

st.markdown("""
<style>
.st-key-web_toggle_btn button,
.st-key-web_toggle_btn button:hover,
.st-key-web_toggle_btn button:focus,
.st-key-web_toggle_btn button:focus:not(:active),
.st-key-web_toggle_btn button:active {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    width: 60px !important;
    height: 60px !important;
    min-width: 60px !important;
    min-height: 60px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.st-key-web_toggle_btn button:hover {
    filter: brightness(1.3);
    cursor: pointer;
}
/* NEW: Streamlit nests the button's label text in an inner element (e.g. a
<p> tag) that carries its own font-size — setting font-size on the outer
<button> alone doesn't override that. Targeting every descendant with `*`
is the documented fix for this exact Streamlit quirk. */
.st-key-web_toggle_btn button,
.st-key-web_toggle_btn button * {
    font-size: 1.35rem !important;
    line-height: 1 !important;
}
</style>
""", unsafe_allow_html=True)

# Plain st.button (no container/key-class dependency) — works on every Streamlit version.
if st.button("🌐", key="web_toggle_btn", help="Toggle live web search"):
    st.session_state.web_search_enabled = not st.session_state.web_search_enabled
    st.rerun()

_web_toggle_on_js = "true" if st.session_state.web_search_enabled else "false"

st.iframe(
    """
    <style>html, body { margin: 0; padding: 0; height: 0; overflow: hidden; }</style>
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

            if (enabled) {
                toggleBtn.style.setProperty('color', '#00A86B');
            } else {
                toggleBtn.style.setProperty('color', '#9CA39C');
            }

            const wrapper = toggleBtn.closest('div[data-testid="stElementContainer"], div.element-container') || toggleBtn;
            wrapper.style.setProperty('position', 'fixed', 'important');
            wrapper.style.setProperty('width', 'auto', 'important');
            wrapper.style.setProperty('z-index', '1000', 'important');

            const chatInput = parentDoc.querySelector('[data-testid="stChatInput"]');
            if (chatInput) {
                const rect = chatInput.getBoundingClientRect();
                wrapper.style.setProperty('top', (rect.top + rect.height / 2 - 28.2) + 'px', 'important');
                wrapper.style.setProperty('left', (rect.left - 50) + 'px', 'important');
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
    height=1,
)

if user_input:
    st.session_state.chips_used = True

# REPLACE your chips block with this:
if not st.session_state.chips_used:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-title">Where should we begin?</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:38px;'></div>", unsafe_allow_html=True)

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

current_messages = st.session_state.chats[st.session_state.current_chat_id]["messages"]
# ---- END NEW ----

# ---- STEP 3: DISPLAY MESSAGES ----
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---- STEP 4: PROCESS MESSAGE ----
if not user_input and not pending:
    st.stop()

if not user_input and pending:
    user_input = pending

current_messages.append({"role": "user", "content": user_input})
save_message(st.session_state.user_id, st.session_state.current_chat_id, "user", user_input)

# ----  auto-title the chat from its first message, ChatGPT-style ----
active_chat = st.session_state.chats[st.session_state.current_chat_id]
if active_chat["title"] == "New Chat":
    active_chat["title"] = _derive_chat_title(current_messages)

with st.chat_message("user"):
    st.markdown(user_input)

with st.chat_message("assistant"):
    placeholder = st.empty()
    full_response = ""

    # ----  always give the model the real current date/time ----
    effective_system_prompt = st.session_state.system_prompt + (
        f"\n\nCurrent date and time: {get_current_datetime_str()}. "
        "Always use this exact date/time if the user asks what today's date, "
        "the day of the week, or the current time is — never guess or rely on "
        "your training cutoff for this."
    )

    # ----  Decide whether to run a live web search (toggle OR auto-detect) ----
    use_web_search = st.session_state.web_search_enabled or needs_web_search(user_input)
    web_context, web_sources = "", []
    if use_web_search:
        with st.spinner("Searching the web..."):
            web_context, web_sources = tavily_web_search(user_input)
   

    # ----  PRIORITY ROUTING — an uploaded image always routes through Gemini,
    # since Gemini can take the image + PDF context + web context + question all
    # in one native multimodal call. This single branch covers every "image
    # present" combination from the routing spec (image alone, image+PDF,
    # image+web, image+PDF+web). ----
    if selected_image and selected_image in current_image_store:
        img_data = current_image_store[selected_image]

        extra_context = ""
        if selected_pdf and selected_pdf in current_pdf_store:
            store = current_pdf_store[selected_pdf]
            relevant = search_chunks(user_input, store["chunks"], store["embeddings"])
            extra_context += "Document context:\n" + "\n\n".join(relevant)
        if web_context:
            extra_context += ("\n\n" if extra_context else "") + f"Live web results:\n{web_context}"

        with st.spinner("Looking at the image..."):
            full_response = gemini_vision_answer(
                user_input,
                img_data["bytes"],
                img_data["mime_type"],
                effective_system_prompt,
                placeholder,
                extra_context,
                temperature=st.session_state.temperature,
                top_p=st.session_state.top_p,
                max_output_tokens=st.session_state.num_predict,
            )
    
    else:
        groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

        if selected_pdf and selected_pdf in current_pdf_store:
            store = current_pdf_store[selected_pdf]
            relevant = search_chunks(user_input, store["chunks"], store["embeddings"])
            context = "\n\n".join(relevant)
            # ----  merge live web results into the RAG context, if any ----
            if web_context:
                context += f"\n\nLive web results:\n{web_context}"
            
            rag_prompt = f"Use this context to answer:\n\n{context}\n\nQuestion: {user_input}"
            messages_to_send = (
                [{"role": "system", "content": effective_system_prompt}]
                + current_messages[:-1]
                + [{"role": "user", "content": rag_prompt}]
            )
        # ----  web-search-only branch (no PDF selected) ----
        elif web_context:
            web_prompt = f"Use these live web search results to answer accurately:\n\n{web_context}\n\nQuestion: {user_input}"
            messages_to_send = (
                [{"role": "system", "content": effective_system_prompt}]
                + current_messages[:-1]
                + [{"role": "user", "content": web_prompt}]
            )
        # ----  NEW ----
        else:
            messages_to_send = (
                [{"role": "system", "content": effective_system_prompt}]
                + current_messages
            )

        stream = groq_client.chat.completions.create(
            model=selected_model,
            messages=messages_to_send,
            stream=True,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
            max_tokens=st.session_state.num_predict,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

    # ----  show sources when a web search was actually used ----
    if web_sources:
        sources_md = "\n\n---\n**🌐 Sources:**\n" + "\n".join(
            [f"- [{s['title']}]({s['url']})" for s in web_sources if s.get("url")]
        )
        full_response += sources_md
        placeholder.markdown(full_response)
    

current_messages.append({"role": "assistant", "content": full_response})
save_message(st.session_state.user_id, st.session_state.current_chat_id, "assistant", full_response)
