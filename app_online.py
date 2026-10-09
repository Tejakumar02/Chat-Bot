import html
import hmac
import json
import re
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import fitz
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from supabase import create_client

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ---- PAGE CONFIG (must be first) ----
st.set_page_config(
    page_title="Greeny-AI",
    page_icon="☘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- SETTINGS ----
# FIX: qwen/qwen3.6-27b was shut down by Groq on 2026-09-14 (replacement: qwen/qwen3.8-27b).
MODEL_OPTIONS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
]
# Vision model can now be switched from secrets.toml without a code change.
GEMINI_MODEL = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
# Gemini counts "thinking" tokens inside max_output_tokens, so thinking gets its own budget
# on top of the admin "Max response tokens" (otherwise answers come back empty/truncated).
GEMINI_THINKING_BUDGET = 1024

# all-MiniLM-L6-v2 truncates input after 256 word pieces, so chunks must stay below that
# (~180 words) or most of each chunk is never embedded.
CHUNK_WORDS = 180
CHUNK_OVERLAP = 30
TOP_K_CHUNKS = 5

# Groq free tier allows 8K tokens/minute per model; sending the whole chat every turn
# eventually fails with "request too large". Keep only the most recent history.
HISTORY_MAX_MESSAGES = 20
HISTORY_CHAR_BUDGET = 12_000

MAX_IMAGE_BYTES = 15 * 1024 * 1024   # Gemini inline request limit is 20 MB (base64 adds ~33%)
DEFAULT_TIMEZONE = "Asia/Kolkata"
UID_COOKIE = "greeny_uid"
NONE_OPTION = "— None (off) —"
SOURCES_MARKER = "\n\n---\n**🌐 Sources:**\n"
THINKING_MD = "_Thinking…_"

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
    --danger:        #e05252;
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

/* ── SIDEBAR BUTTONS ── */
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

/* FIX: was ".stButton:first-of-type", which matches EVERY sidebar button
   (each button is the first div inside its own element container). */
[data-testid="stSidebar"] .st-key-new_chat_btn {
    margin-top: 12px;
}

/* ── CHAT LIST (sidebar) ── */
[data-testid="stSidebar"] [class*="st-key-select_"] .stButton > button {
    justify-content: flex-start !important;
    text-align: left !important;
    padding: 0.45rem 0.8rem !important;
}
[data-testid="stSidebar"] [class*="st-key-select_"] button > div {
    min-width: 0 !important;
    overflow: hidden !important;
}
[data-testid="stSidebar"] [class*="st-key-select_"] button p {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] [class*="st-key-del_"] .stButton > button,
[data-testid="stSidebar"] [class*="st-key-cancel_del_"] .stButton > button,
[data-testid="stSidebar"] [class*="st-key-del_"] .stButton > button:hover,
[data-testid="stSidebar"] [class*="st-key-cancel_del_"] .stButton > button:hover {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0.3rem !important;
    min-height: 0 !important;
}
[data-testid="stSidebar"] [class*="st-key-del_"] button span,
[data-testid="stSidebar"] [class*="st-key-cancel_del_"] button span {
    font-size: 1.15rem !important;
    color: var(--text-muted) !important;
    transition: color 0.15s ease !important;
}
[data-testid="stSidebar"] [class*="st-key-del_"] button:hover span {
    color: var(--danger) !important;
}
[data-testid="stSidebar"] [class*="st-key-cancel_del_"] button:hover span {
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] [class*="st-key-confirm_del_"] .stButton > button {
    border-color: rgba(224, 82, 82, 0.6) !important;
    background: rgba(224, 82, 82, 0.08) !important;
}
[data-testid="stSidebar"] [class*="st-key-confirm_del_"] button p {
    color: var(--danger) !important;
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
    padding-top: 0 !important;
    padding-right: 2.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2.5rem !important;
    max-width: 860px !important;
}

/* FIX: the chat input lives in stBottomBlockContainer, which does NOT get the
   "block-container" class, so in wide layout it was much wider than the messages. */
[data-testid="stBottomBlockContainer"] {
    max-width: 860px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-left: calc(2.5rem + var(--greeny-reserve, 0px)) !important;
    padding-right: 2.5rem !important;
}

[data-testid="stToolbar"] {
    display: block !important;
}
[data-testid="stHeader"] {
    height: auto !important;
    min-height: auto !important;
}

[data-testid="stMain"] .block-container > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

iframe {
    border: none !important;
    display: block !important;
}

/* Helper iframe (web-toggle positioning + cookie) takes no space in the layout. */
.st-key-greeny_bridge {
    position: absolute !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* ── MODEL + STATUS PILLS ── */
.model-pill {
    position: fixed;
    top: 80px;
    right: 85px;
    z-index: 999;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-pill);
    padding: 6px 14px;
    color: var(--accent-light);
    font-size: 0.75rem;
    font-weight: 600;
}

.model-pill::before {
    content: '';
    width: 6px; height: 6px;
    background: var(--accent-light);
    border-radius: 50%;
    box-shadow: 0 0 6px var(--accent-light);
}

@media (max-width: 640px) {
    .model-pill { top: 64px; right: 16px; }
}

/* ── CHAT MESSAGES ──
   FIX: Streamlit's avatar test-ids are stChatMessageAvatarUser / stChatMessageAvatarAssistant.
   The old "chatAvatarIcon-user/assistant" selectors matched nothing, so no bubble styling applied. */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.2rem 0 !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    flex: 0 1 auto !important;
    margin: auto 0 !important;
    max-width: 78% !important;
    background: linear-gradient(135deg, var(--accent), #004d33) !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 0.75rem 1rem !important;
    color: #fff !important;
    box-shadow: 0 2px 12px var(--accent-glow) !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
    margin: auto 0 !important;   /* Streamlit's margin:auto would centre the capped-width bubble */
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 16px 16px 16px 4px !important;
    padding: 0.75rem 1rem !important;
    color: var(--text-primary) !important;
    max-width: 85% !important;
}

[data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(135deg, var(--accent), var(--accent-light)) !important;
    border-radius: 10px !important;
}

[data-testid="stChatMessageAvatarAssistant"] {
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

/* ── WEB SEARCH TOGGLE (pinned left of the chat input by the helper script) ──
   UX: uses a Material icon instead of the 🌐 emoji, because colour emoji ignore CSS
   "color", so the old ON/OFF colour change was invisible. */
.st-key-web_toggle_btn {
    position: fixed !important;
    top: var(--greeny-toggle-top, auto) !important;
    left: var(--greeny-toggle-left, 20px) !important;
    bottom: var(--greeny-toggle-bottom, 22px) !important;
    width: 40px !important;
    z-index: 1000 !important;
}
.st-key-web_toggle_btn button {
    width: 40px !important;
    height: 40px !important;
    min-height: 40px !important;
    padding: 0 !important;
    border-radius: 50% !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    box-shadow: none !important;
    color: var(--text-muted) !important;
    transition: color 0.2s ease, background 0.2s ease, box-shadow 0.2s ease !important;
}
.st-key-web_toggle_btn button span {
    font-size: 1.45rem !important;
    color: inherit !important;
}
.st-key-web_toggle_btn button:hover {
    color: var(--text-secondary) !important;
    background: var(--accent-subtle) !important;
}

/* ── EMPTY STATE — centered like ChatGPT ── */
.welcome-wrapper {
    margin-top: 45px;
    margin-bottom: -20px;
    text-align: center;
}

.empty-title {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text-primary);
}

/* FIX: chip styling is scoped to its own container. The old rules targeted every
   column block in the main area, contradicted each other (white vs grey text,
   zero padding) and relied on a script tag in st.markdown, which never executes. */
.st-key-welcome_chips .stButton > button {
    background: var(--accent) !important;
    border: none !important;
    border-radius: var(--radius-pill) !important;
    padding: 6px 14px !important;
    min-height: 0 !important;
    box-shadow: none !important;
    white-space: nowrap !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease !important;
}
.st-key-welcome_chips .stButton > button p {
    color: #ffffff !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}
.st-key-welcome_chips .stButton > button:hover {
    background: #004d33 !important;
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

/* ── SIDEBAR CLOSE / OPEN BUTTONS ── */
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
    opacity: 1 !important;
    visibility: visible !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button,
[data-testid="stExpandSidebarButton"] {
    opacity: 1 !important;
    visibility: visible !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    transition: all 0.2s ease !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stExpandSidebarButton"]:hover {
    background: var(--accent-subtle) !important;
    border-color: var(--accent-light) !important;
}

/* FIX: the expand control's test-id is stExpandSidebarButton ("collapsedControl" no longer exists). */
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button svg,
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button span,
[data-testid="stExpandSidebarButton"] svg,
[data-testid="stExpandSidebarButton"] span {
    opacity: 1 !important;
    fill: var(--accent-light) !important;
    color: var(--accent-light) !important;
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

/* ── STATUS PILLS (RAG / web / vision) ── */
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
    margin: 0 6px 0.8rem 0;
}

</style>
""", unsafe_allow_html=True)


# ---- CACHED CLIENTS ----
# PERF: these used to be re-created on every rerun / every message.
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_supabase():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


@st.cache_resource
def get_tavily_client():
    return TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])


embedder = load_embedder()
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD")


# ---- SMALL HELPERS ----
def esc(text):
    """Escape user-controlled text (file names, titles) before putting it in raw HTML."""
    return html.escape(str(text), quote=True)


def flash(message, icon=":material/info:"):
    """Queue a toast. Works from callbacks and survives the rerun that follows them."""
    st.session_state.setdefault("_flash", []).append((message, icon))


def show_flashes():
    for message, icon in st.session_state.pop("_flash", []):
        st.toast(message, icon=icon)


def _empty_chat(created, title="New Chat"):
    return {
        "title": title,
        "created": created,
        "updated": created,
        "messages": [],
        "pdf_store": {},
        "image_store": {},
    }


def current_chat():
    return st.session_state.chats[st.session_state.current_chat_id]


def _valid_uuid(value):
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        return None


def strip_sources(content):
    """Remove the '🌐 Sources' footer before re-sending a reply to the model as history."""
    return content.split(SOURCES_MARKER, 1)[0]


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def clean_reply(text):
    """Drop reasoning blocks some models emit inline (complete or still streaming)."""
    text = _THINK_BLOCK_RE.sub("", text)
    open_idx = text.lower().find("<think>")
    if open_idx != -1:
        text = text[:open_idx]
    return text.lstrip()


# ---- DATABASE (failures no longer crash the app) ----
def save_message(user_id, chat_id, role, content):
    try:
        get_supabase().table("messages").insert({
            "user_id": user_id,
            "chat_id": chat_id,
            "role": role,
            "content": content
        }).execute()
        return True
    except Exception as e:
        flash(f"Couldn't save this message to history: {e}", ":material/cloud_off:")
        return False


def _derive_chat_title(messages):
    """Auto-generates a chat title from the first user message, ChatGPT-style."""
    for m in messages:
        if m["role"] == "user":
            text = m["content"].strip().replace("\n", " ")
            return text[:40] + ("…" if len(text) > 40 else "")
    return "New Chat"


def load_chats(user_id):
    """
    Loads every saved message for this user and groups it into chats by chat_id.
    Rows with no chat_id (saved before chats existed) are grouped under "legacy".
    FIX: pages through the results — Supabase returns at most 1000 rows per request,
    so heavy users silently lost their most recent messages.
    """
    rows, start, page_size = [], 0, 1000
    while True:
        result = (
            get_supabase().table("messages")
            .select("id, chat_id, role, content")
            .eq("user_id", user_id)
            .order("id")
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = result.data or []
        rows.extend(batch)
        if not batch:
            break
        start += len(batch)

    chats = {}
    for position, row in enumerate(rows):
        cid = row.get("chat_id") or "legacy"
        if cid not in chats:
            chats[cid] = _empty_chat(created=position)
        chats[cid]["messages"].append({"role": row["role"], "content": row["content"] or ""})
        chats[cid]["updated"] = position
    for cid, chat in chats.items():
        chat["title"] = "Previous Chat" if cid == "legacy" else _derive_chat_title(chat["messages"])
    return chats


def delete_chat_messages(user_id, chat_id):
    query = get_supabase().table("messages").delete().eq("user_id", user_id)
    if chat_id == "legacy":
        query = query.is_("chat_id", "null")
    else:
        query = query.eq("chat_id", chat_id)
    query.execute()


# ---- CHAT SESSION MANAGEMENT (all used as on_click callbacks) ----
def _reset_attachment_selection():
    st.session_state["pdf_selectbox"] = NONE_OPTION
    st.session_state["image_selectbox"] = NONE_OPTION


def new_chat():
    # UX: don't pile up empty "New Chat" entries — reuse the current one if it's still empty.
    st.session_state.failed_turn = None
    existing = st.session_state.chats.get(st.session_state.get("current_chat_id"))
    if existing is not None and not existing["messages"] \
            and not existing["pdf_store"] and not existing["image_store"]:
        st.session_state.chips_used = False
        return
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = _empty_chat(created=time.time())
    st.session_state.current_chat_id = new_id
    st.session_state.chips_used = False  # show the empty-state welcome screen again
    st.session_state.confirm_delete = None
    _reset_attachment_selection()


def select_chat(chat_id):
    st.session_state.current_chat_id = chat_id
    st.session_state.chips_used = len(st.session_state.chats[chat_id]["messages"]) > 0
    st.session_state.confirm_delete = None
    # each chat has its own pdf/image store, so don't carry the selection over
    _reset_attachment_selection()


def ask_delete(chat_id):
    st.session_state.confirm_delete = chat_id


def cancel_delete():
    st.session_state.confirm_delete = None


def delete_chat(chat_id):
    st.session_state.confirm_delete = None
    try:
        delete_chat_messages(st.session_state.user_id, chat_id)
    except Exception as e:
        flash(f"Couldn't delete the chat: {e}", ":material/error:")
        return
    st.session_state.chats.pop(chat_id, None)
    if st.session_state.current_chat_id == chat_id or not st.session_state.chats:
        new_chat()
    flash("Chat deleted", ":material/delete:")


# ---- PDF UTILS ----
def load_pdf(data):
    with fitz.open(stream=data, filetype="pdf") as doc:
        if doc.needs_pass:
            raise ValueError("it is password-protected")
        return "\n".join(page.get_text() for page in doc)


def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return chunks


def search_chunks(query, chunks, embeddings, n=TOP_K_CHUNKS):
    if not chunks:
        return []
    query_emb = np.asarray(embedder.encode([query], normalize_embeddings=True))
    scores = (np.asarray(embeddings) @ query_emb.T).ravel()
    top_indices = np.argsort(scores)[::-1][:min(n, len(chunks))]
    return [chunks[i] for i in top_indices]


# ---- DATE/TIME + WEB SEARCH UTILS ----
def get_current_datetime_str():
    """Real current date/time in the user's browser timezone (falls back to IST)."""
    tz_name = None
    try:
        tz_name = st.context.timezone
    except Exception:
        pass
    try:
        tz = ZoneInfo(tz_name or DEFAULT_TIMEZONE)
    except (ZoneInfoNotFoundError, ValueError):
        tz = ZoneInfo(DEFAULT_TIMEZONE)
    return datetime.now(tz).strftime("%A, %B %d, %Y - %I:%M %p %Z")


# FIX: the old check used plain substring matching, so "know" matched "now",
# "underscore" matched "score", "deliver" matched "live", "renews" matched "news" —
# triggering paid, slow web searches for ordinary questions. Whole words/phrases only,
# and words that are ambiguous on their own ("now", "live", "update", "current") only in
# phrases that really ask for fresh information. The 🌐 toggle still forces a search.
_WEB_SEARCH_RE = re.compile(
    r"\b(?:"
    r"today|tonight|yesterday|latest|recent|recently|breaking|news|headlines?|"
    r"weather|forecast|right now|as of now|happening now|as of today|"
    r"this (?:week|month|year)|"
    r"live (?:score|scores|updates?|news)|scores? (?:of|for) (?:the|today)|"
    r"stock price|share price|exchange rate|"
    r"current (?:events?|news|price|prices|weather|president|prime minister|ceo|status)|"
    r"what(?:'s| is) the (?:date|time)|what date|what time is it|what day is it|"
    r"who is the (?:current|president|prime minister|ceo)"
    r")\b",
    re.IGNORECASE,
)


def needs_web_search(query):
    """Lightweight auto-detect for queries that likely need live info."""
    return bool(_WEB_SEARCH_RE.search(query or ""))


def tavily_web_search(query, max_results=5):
    """Returns (context_text, sources_list, error_message_or_None). Never raises."""
    if not TAVILY_AVAILABLE:
        return "", [], "the `tavily-python` package isn't installed"
    if "TAVILY_API_KEY" not in st.secrets:
        return "", [], "no `TAVILY_API_KEY` in secrets.toml"
    try:
        response = get_tavily_client().search(query=query, max_results=max_results)
        context_parts, sources = [], []
        for r in response.get("results", []):
            title = r.get("title", "Source")
            context_parts.append(f"Source: {title}\n{r.get('content', '')}")
            sources.append({"title": title, "url": r.get("url", "")})
        return "\n\n".join(context_parts), sources, None
    except Exception as e:
        return "", [], str(e)


def build_history(messages, max_messages=HISTORY_MAX_MESSAGES, char_budget=HISTORY_CHAR_BUDGET):
    """Most recent turns that fit the budget, without the 'Sources' footers."""
    picked, used = [], 0
    for m in reversed(messages[-max_messages:]):
        content = strip_sources(m["content"]).strip()
        if not content:
            continue
        if picked and used + len(content) > char_budget:
            break
        picked.append({"role": m["role"], "content": content})
        used += len(content)
    picked.reverse()
    while picked and picked[0]["role"] != "user":
        picked.pop(0)
    return picked


def friendly_error(e):
    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    text = str(e)
    low = text.lower()
    if isinstance(e, KeyError):
        return f"🔑 Missing secret {text} in secrets.toml."
    if status == 429 or "rate limit" in low:
        return "⏳ Rate limit reached for this model. Wait a few seconds and retry, or pick another model in the sidebar."
    if status == 413 or "too large" in low:
        return "📏 This request is too large for the model's token limit. Start a new chat, or turn off the document/web search, and retry."
    if status in (401, 403):
        return "🔑 The API key was rejected. Check the key in secrets.toml."
    if status == 404 or "decommissioned" in low or "no longer available" in low:
        return f"🚫 This model isn't available any more. Pick another model in the sidebar. ({text[:200]})"
    return f"⚠️ Something went wrong while generating the reply: {text[:300]}"


# ---- MODEL CALLS ----
def stream_groq(model, messages, placeholder, acc):
    stream = get_groq_client().chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=st.session_state.temperature,
        top_p=st.session_state.top_p,
        max_completion_tokens=st.session_state.num_predict,  # FIX: max_tokens is deprecated
    )
    raw, finish, last_draw = "", None, 0.0
    placeholder.markdown(THINKING_MD)
    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        piece = getattr(choice.delta, "content", None) if choice.delta else None
        if piece:
            raw += piece
            acc["text"] = clean_reply(raw)
            now = time.monotonic()
            if now - last_draw > 0.05:  # throttle redraws
                placeholder.markdown((acc["text"] + "▌") if acc["text"] else THINKING_MD)
                last_draw = now
        if choice.finish_reason:
            finish = choice.finish_reason
    return clean_reply(raw), finish


def stream_gemini(prompt_text, image_bytes, mime_type, system_instruction, history, placeholder, acc):
    """Image (+ optional PDF/web context) answered by Gemini, now WITH chat history."""
    if not GEMINI_AVAILABLE:
        raise RuntimeError("Vision support isn't installed. Add `google-genai` to requirements.txt.")
    if "GEMINI_API_KEY" not in st.secrets:
        raise RuntimeError("Vision support needs a `GEMINI_API_KEY` in secrets.toml.")

    contents = [
        genai_types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[genai_types.Part.from_text(text=m["content"])],
        )
        for m in history
    ]
    contents.append(genai_types.Content(role="user", parts=[
        genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        genai_types.Part.from_text(text=prompt_text),
    ]))
    stream = get_gemini_client().models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=st.session_state.temperature,
            top_p=st.session_state.top_p,
            max_output_tokens=st.session_state.num_predict + GEMINI_THINKING_BUDGET,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGET),
        ),
    )
    text, finish, last_draw = "", None, 0.0
    placeholder.markdown("_🖼️ Looking at the image…_")
    for chunk in stream:
        piece = chunk.text
        if piece:
            text += piece
            acc["text"] = text
            now = time.monotonic()
            if now - last_draw > 0.05:
                placeholder.markdown(text + "▌")
                last_draw = now
        candidates = getattr(chunk, "candidates", None)
        if candidates and getattr(candidates[0], "finish_reason", None):
            reason = candidates[0].finish_reason
            finish = getattr(reason, "name", str(reason))
    if finish == "MAX_TOKENS":
        finish = "length"
    return text, finish


# ---- ATTACHMENTS ----
# strict either/or — picking one mode auto-deactivates the other
def _on_pdf_selectbox_change():
    if st.session_state.get("pdf_selectbox") not in (None, NONE_OPTION):
        st.session_state["image_selectbox"] = NONE_OPTION


def _on_image_selectbox_change():
    if st.session_state.get("image_selectbox") not in (None, NONE_OPTION):
        st.session_state["pdf_selectbox"] = NONE_OPTION


def _handle_attached_file(uploaded_file):
    chat = current_chat()
    name = uploaded_file.name
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    data = uploaded_file.getvalue()

    if ext == "pdf":
        existing = chat["pdf_store"].get(name)
        if existing is None or existing.get("size") != len(data):  # re-index a changed file
            try:
                text = load_pdf(data)
            except Exception as e:
                flash(f"Couldn't read “{name}”: {e}", ":material/error:")
                return
            chunks = chunk_text(text)
            if not chunks:
                # FIX: a scanned/image-only PDF has no text; it used to crash on the first question.
                flash(f"“{name}” has no selectable text (it may be a scanned PDF). "
                      "Attach a screenshot of the page as an image instead.", ":material/error:")
                return
            with st.spinner(f"Reading {name}…"):
                embeddings = np.asarray(embedder.encode(chunks, normalize_embeddings=True))
            chat["pdf_store"][name] = {"chunks": chunks, "embeddings": embeddings, "size": len(data)}
        st.session_state["pdf_selectbox"] = name
        st.session_state["image_selectbox"] = NONE_OPTION
        flash(f"“{name}” is ready — ask me anything about it.", ":material/description:")
        return "pdf"
    else:
        if len(data) > MAX_IMAGE_BYTES:
            flash(f"“{name}” is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB. Please attach a smaller image.",
                  ":material/error:")
            return
        chat["image_store"][name] = {
            "bytes": data,
            "mime_type": uploaded_file.type or f"image/{'jpeg' if ext == 'jpg' else ext}",
        }
        st.session_state["image_selectbox"] = name
        st.session_state["pdf_selectbox"] = NONE_OPTION
        flash(f"“{name}” is ready — ask me anything about it.", ":material/description:")
        return "pdf"


def clear_active_document():
    name = st.session_state.get("pdf_selectbox")
    current_chat()["pdf_store"].pop(name, None)
    st.session_state["pdf_selectbox"] = NONE_OPTION


# ---- CHIP / RETRY CALLBACKS ----
def prefill_input(text):
    # UX: incomplete prompts ("Write me a Python function that") are now put in the chat
    # box for the user to finish, instead of being sent to the model as-is.
    st.session_state.chat_input = text
    st.session_state.focus_chat_input = True


def chip_summarize():
    if st.session_state.get("pdf_selectbox") not in (None, NONE_OPTION):
        st.session_state.pending_message = "Please summarize the uploaded document for me."
        st.session_state.chips_used = True
    elif st.session_state.get("image_selectbox") not in (None, NONE_OPTION):
        st.session_state.pending_message = "Please describe and summarize this image."
        st.session_state.chips_used = True
    else:
        flash("Attach a PDF first (📎 in the chat box), then press Summarize.", ":material/attach_file:")


def retry_failed():
    failed = st.session_state.get("failed_turn")
    if failed:
        st.session_state.pending_message = failed["text"]
    st.session_state.failed_turn = None


def toggle_web_search():
    st.session_state.web_search_enabled = not st.session_state.web_search_enabled


def lock_admin():
    st.session_state.admin_unlocked = False


def _persist(widget_key, state_key):
    st.session_state[state_key] = st.session_state[widget_key]


# ---- SESSION STATE ----
defaults = {
    "pending_message": "",
    "chips_used": False,
    "web_search_enabled": False,
    "admin_unlocked": False,
    "system_prompt": "You are a helpful assistant.",
    "temperature": 0.7,
    "top_p": 1.0,
    "num_predict": 1024,
    "confirm_delete": None,
    "failed_turn": None,
    "focus_chat_input": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# FIX: user_id used to be a brand-new uuid4() for every browser session, so load_chats()
# never found anything after a refresh. It's now remembered in a browser cookie.
if "user_id" not in st.session_state:
    cookie_uid = None
    try:
        cookie_uid = _valid_uuid(st.context.cookies.get(UID_COOKIE))
    except Exception:
        pass
    st.session_state.user_id = cookie_uid or str(uuid.uuid4())
    st.session_state.uid_cookie_needed = cookie_uid is None

if "chats" not in st.session_state:
    try:
        st.session_state.chats = load_chats(st.session_state.user_id)
    except Exception as e:
        st.session_state.chats = {}
        flash(f"Couldn't load your chat history: {e}", ":material/cloud_off:")

if "current_chat_id" not in st.session_state or \
        st.session_state.current_chat_id not in st.session_state.chats:
    if st.session_state.chats:
        # Most recently active chat becomes active on load; show its messages, not the welcome screen.
        st.session_state.current_chat_id = max(
            st.session_state.chats.items(), key=lambda kv: kv[1]["updated"]
        )[0]
        st.session_state.chips_used = len(current_chat()["messages"]) > 0
    else:
        new_chat()

if st.session_state.get("selected_model") not in MODEL_OPTIONS:
    st.session_state.selected_model = MODEL_OPTIONS[0]

# ---- Capture chat input BEFORE sidebar (attachments update the sidebar selectors) ----
chat_value = st.chat_input(
    "Ask anything...",
    key="chat_input",
    accept_file=True,
    file_type=["pdf", "png", "jpg", "jpeg", "webp"],
)

user_input = ""
if chat_value:
    attached = [_handle_attached_file(f) for f in chat_value.files]
    user_input = (chat_value.text or "").strip()
    # FIX: a file sent with no text used to hit st.stop() (no message = no reply).
    # Use a sensible default question for the file that was just attached.
    if not user_input and "image" in attached:
        user_input = "Describe this image in detail."
    elif not user_input and "pdf" in attached:
        user_input = "Please summarize the uploaded document for me."
    st.session_state.failed_turn = None  # a new message replaces an unanswered one

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

    st.markdown('<div class="sidebar-label">Model</div>', unsafe_allow_html=True)
    st.selectbox("Model", MODEL_OPTIONS, key="selected_model", label_visibility="collapsed")

    st.button("New chat", icon=":material/add:", key="new_chat_btn", width="stretch", on_click=new_chat)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Your Chats")

    chat_items = sorted(
        st.session_state.chats.items(), key=lambda kv: kv[1]["updated"], reverse=True
    )
    for chat_id, chat in chat_items:
        col_select, col_delete = st.columns([4, 1], vertical_alignment="center")
        if st.session_state.confirm_delete == chat_id:
            # UX: deleting is permanent (it removes rows from the database), so confirm first.
            col_select.button("Delete this chat?", key=f"confirm_del_{chat_id}", width="stretch",
                              help="Click to delete permanently", on_click=delete_chat, args=(chat_id,))
            col_delete.button("", icon=":material/close:", key=f"cancel_del_{chat_id}",
                              help="Cancel", on_click=cancel_delete)
        else:
            col_select.button(chat["title"], key=f"select_{chat_id}", width="stretch",
                              help=chat["title"], on_click=select_chat, args=(chat_id,))
            col_delete.button("", icon=":material/delete:", key=f"del_{chat_id}",
                              help="Delete chat", on_click=ask_delete, args=(chat_id,))

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---- Admin Settings (password-gated) ----
    with st.expander("⚙️ Admin settings"):
        if not st.session_state.admin_unlocked:
            st.caption("Locked. Enter the admin password to change these.")
            with st.form("admin_unlock_form", border=False):  # UX: Enter key submits
                pw = st.text_input("Password", type="password")
                unlock = st.form_submit_button("Unlock")
            if unlock:
                if not ADMIN_PASSWORD:
                    st.error("No ADMIN_PASSWORD set in secrets.toml — admin panel can't unlock.")
                elif hmac.compare_digest(pw.encode(), str(ADMIN_PASSWORD).encode()):
                    st.session_state.admin_unlocked = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
        else:
            # Widgets get their own keys and copy into the persistent keys on change, so the
            # values survive locking the panel and the widgets aren't re-created every rerun.
            for widget_key, state_key in (("_w_system_prompt", "system_prompt"),
                                          ("_w_temperature", "temperature"),
                                          ("_w_top_p", "top_p"),
                                          ("_w_num_predict", "num_predict")):
                st.session_state[widget_key] = st.session_state[state_key]

            st.markdown('<div class="sidebar-label">System Prompt</div>', unsafe_allow_html=True)
            st.text_area("System prompt", key="_w_system_prompt", height=90, label_visibility="collapsed",
                         on_change=_persist, args=("_w_system_prompt", "system_prompt"))
            st.slider("Temperature", 0.0, 2.0, step=0.05, key="_w_temperature",
                      on_change=_persist, args=("_w_temperature", "temperature"),
                      help="Higher = more random/creative. Lower = more focused/deterministic.")
            st.slider("Top-p", 0.05, 1.0, step=0.05, key="_w_top_p",
                      on_change=_persist, args=("_w_top_p", "top_p"))
            st.slider("Max response tokens", 128, 4096, step=128, key="_w_num_predict",
                      on_change=_persist, args=("_w_num_predict", "num_predict"),
                      help="Reasoning models spend part of this budget thinking. "
                           "Raise it if replies get cut off.")

            if st.session_state.get("pdf_selectbox") not in (None, NONE_OPTION):
                st.button("Clear active document", on_click=clear_active_document)

            st.button("🔒 Lock settings", on_click=lock_admin)

    st.markdown("<hr>", unsafe_allow_html=True)

    current_pdf_store = current_chat()["pdf_store"]
    current_image_store = current_chat()["image_store"]

    selected_pdf = None
    if current_pdf_store:
        st.markdown('<div class="sidebar-label">Active Document</div>', unsafe_allow_html=True)
        for doc_name in current_pdf_store.keys():
            st.markdown(f"""
            <div class="doc-badge">
                <span class="doc-badge-icon">📄</span>
                {esc(doc_name[:28])}{'...' if len(doc_name) > 28 else ''}
            </div>
            """, unsafe_allow_html=True)
        pdf_choice = st.selectbox(
            "Search in:", [NONE_OPTION] + list(current_pdf_store.keys()),
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
                {esc(img_name[:28])}{'...' if len(img_name) > 28 else ''}
            </div>
            """, unsafe_allow_html=True)
        image_choice = st.selectbox(
            "Analyze:", [NONE_OPTION] + list(current_image_store.keys()),
            label_visibility="collapsed", key="image_selectbox",
            on_change=_on_image_selectbox_change
        )
        selected_image = None if image_choice == NONE_OPTION else image_choice
        if selected_image:
            st.caption(f"Images are answered by {GEMINI_MODEL}.")

    st.markdown("<hr>", unsafe_allow_html=True)

    current_messages = current_chat()["messages"]
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

selected_model = st.session_state.selected_model

# ---- Per-run CSS: active chat highlight + web toggle state (replaces the JS polling) ----
active_css = f"""
[data-testid="stSidebar"] .st-key-select_{st.session_state.current_chat_id} .stButton > button {{
    background: var(--accent-subtle) !important;
    border-color: var(--accent) !important;
}}
[data-testid="stSidebar"] .st-key-select_{st.session_state.current_chat_id} button p {{
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}}
"""
if st.session_state.web_search_enabled:
    active_css += """
.st-key-web_toggle_btn button {
    color: var(--accent-light) !important;
    background: var(--accent-subtle) !important;
    border-color: var(--border) !important;
    box-shadow: 0 0 10px var(--accent-glow) !important;
}
"""
st.markdown(f"<style>{active_css}</style>", unsafe_allow_html=True)

# ---- Model pill + status pills (plain HTML; the old version re-created the pill via JS every second) ----
pill_model = GEMINI_MODEL if selected_image else selected_model  # show the model that will really answer
pills = [f'<div class="model-pill">{esc(pill_model)}</div>']
if selected_pdf:
    pills.append(f'<div class="rag-pill">⚡ RAG active — querying <strong>{esc(selected_pdf[:30])}</strong></div>')
if st.session_state.web_search_enabled:
    pills.append('<div class="rag-pill">🌐 Live web search — ON</div>')
if selected_image:
    pills.append(f'<div class="rag-pill">🖼️ Vision active — analyzing <strong>{esc(selected_image[:30])}</strong></div>')
st.markdown("".join(pills), unsafe_allow_html=True)

# ---- Web search toggle (positioned next to the chat box by the helper script below) ----
st.button(
    "", icon=":material/language:", key="web_toggle_btn", on_click=toggle_web_search,
    help="Live web search is ON — click to turn off" if st.session_state.web_search_enabled
    else "Live web search is OFF — click to turn on",
)

# ---- One small helper iframe (was three, two of them polling every 0.4–1 s) ----
_bridge_cfg = {
    "uid": st.session_state.user_id if st.session_state.get("uid_cookie_needed") else None,
    "cookieName": UID_COOKIE,
    "focus": bool(st.session_state.focus_chat_input),
}
st.session_state.focus_chat_input = False
with st.container(key="greeny_bridge"):
    st.iframe(
        """
        <style>html, body { margin: 0; padding: 0; height: 0; overflow: hidden; }</style>
        <script>
        (function () {
            var CFG = """ + json.dumps(_bridge_cfg) + """;
            var doc, root;
            try { doc = window.parent.document; root = doc.documentElement; } catch (e) { return; }

            // Remember this browser's user id so chat history survives a refresh.
            if (CFG.uid) {
                try {
                    var secure = window.parent.location.protocol === 'https:' ? '; Secure' : '';
                    doc.cookie = CFG.cookieName + '=' + CFG.uid +
                        '; path=/; max-age=31536000; SameSite=Lax' + secure;
                } catch (e) {}
            }

            function cssPx(name) {
                return parseFloat(getComputedStyle(root).getPropertyValue(name)) || 0;
            }

            // Pin the web toggle 8px left of the chat box. If there's no room (narrow screen
            // or open sidebar), reserve space by shifting the chat box right.
            function place() {
                var input = doc.querySelector('[data-testid="stChatInput"]');
                if (!input) return;
                var main = doc.querySelector('[data-testid="stMain"]');
                var minLeft = (main ? main.getBoundingClientRect().left : 0) + 8;
                var r = input.getBoundingClientRect();
                for (var i = 0; i < 4 && r.width; i++) {
                    var reserve = cssPx('--greeny-reserve');
                    var room = r.left - 48 - minLeft;
                    var next = Math.max(0, Math.round(reserve - room));
                    if (Math.abs(next - reserve) < 1) break;
                    root.style.setProperty('--greeny-reserve', next + 'px');
                    r = input.getBoundingClientRect();
                }
                if (!r.width) return;
                root.style.setProperty('--greeny-toggle-top', (r.top + r.height / 2 - 20) + 'px');
                root.style.setProperty('--greeny-toggle-left', (r.left - 48) + 'px');
                root.style.setProperty('--greeny-toggle-bottom', 'auto');
            }
            place();
            setInterval(place, 300);

            if (CFG.focus) {
                setTimeout(function () {
                    var ta = doc.querySelector('[data-testid="stChatInput"] textarea');
                    if (ta) { ta.focus(); var n = ta.value.length; ta.setSelectionRange(n, n); }
                }, 250);
            }
        })();
        </script>
        """,
        height=1,
    )

# ---- STEP 1: CAPTURE PENDING MESSAGE (chips / retry) ----
pending = st.session_state.get("pending_message", "")
if pending:
    st.session_state.pending_message = ""
    st.session_state.chips_used = True

if user_input:
    st.session_state.chips_used = True

# FIX: Streamlit keeps elements from the previous run on screen (faded) until the
# script overwrites their slot or the run ends. The welcome screen used 3 separate
# slots, and the chips' slot was only overwritten after the reply finished streaming,
# so they lingered for the whole reply. One fixed slot is cleared the moment we get here.
welcome_slot = st.empty()
if not st.session_state.chips_used:
    with welcome_slot.container():
        st.markdown("""
        <div class="welcome-wrapper">
            <div class="empty-title">Where should we begin?</div>
        </div>
        """, unsafe_allow_html=True)

        # Space between title and pills
        st.markdown("<div style='height:65px;'></div>", unsafe_allow_html=True)

        with st.container(horizontal=True, horizontal_alignment="center", gap="small", key="welcome_chips"):
            st.button("📄 Summarize", key="chip_summarize", on_click=chip_summarize)
            st.button("💻 Write code", key="chip_code", on_click=prefill_input,
                      args=("Write a Python function that ",))
            st.button("🏖️ Best Places", key="chip_places", on_click=prefill_input,
                      args=("What are the best places to visit in ",))
            st.button("🔍 Compare", key="chip_compare", on_click=prefill_input,
                      args=("Compare and contrast ",))
# ---- STEP 2: DISPLAY MESSAGES ----
current_messages = current_chat()["messages"]
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for note in message.get("notes", ()):  # e.g. "reply cut off" (shown, not saved to DB)
            st.caption(note)

# An unanswered message from a failed request stays visible with a Retry button.
failed = st.session_state.failed_turn
if failed and failed["chat_id"] == st.session_state.current_chat_id and not user_input and not pending:
    with st.chat_message("user"):
        st.markdown(failed["text"])
    with st.chat_message("assistant"):
        st.error(failed["error"])
        st.button("Retry", icon=":material/refresh:", key="retry_btn", on_click=retry_failed)

show_flashes()

# ---- STEP 3: PROCESS MESSAGE ----
if not user_input and not pending:
    st.stop()

if not user_input and pending:
    user_input = pending

chat_id = st.session_state.current_chat_id
active_chat = current_chat()
history = build_history(current_messages)  # previous turns only, trimmed
current_messages.append({"role": "user", "content": user_input})
active_chat["updated"] = time.time()
if active_chat["title"] == "New Chat":
    active_chat["title"] = _derive_chat_title(current_messages)

with st.chat_message("user"):
    st.markdown(user_input)


def _record_failure(error_text):
    """Roll back the unanswered turn and keep it on screen with a Retry button."""
    if current_messages and current_messages[-1] == {"role": "user", "content": user_input}:
        current_messages.pop()
    if not current_messages:
        active_chat["title"] = "New Chat"
    st.session_state.failed_turn = {"chat_id": chat_id, "text": user_input, "error": error_text}


with st.chat_message("assistant"):
    placeholder = st.empty()
    acc = {"text": ""}  # partial reply, kept if the run is interrupted mid-stream
    notes = []

    # always give the model the real current date/time
    effective_system_prompt = st.session_state.system_prompt + (
        f"\n\nCurrent date and time: {get_current_datetime_str()}. "
        "Always use this exact date/time if the user asks what today's date, "
        "the day of the week, or the current time is — never guess or rely on "
        "your training cutoff for this."
    )

    try:
        # Decide whether to run a live web search (toggle OR auto-detect)
        web_requested = st.session_state.web_search_enabled
        web_context, web_sources = "", []
        if web_requested or needs_web_search(user_input):
            placeholder.markdown("_🌐 Searching the web…_")
            web_context, web_sources, web_error = tavily_web_search(user_input)
            if web_error and web_requested:
                notes.append(f"⚠️ Web search unavailable: {web_error}")

        doc_context = ""
        if selected_pdf and selected_pdf in current_pdf_store:
            store = current_pdf_store[selected_pdf]
            doc_context = "\n\n".join(search_chunks(user_input, store["chunks"], store["embeddings"]))

        if selected_image and selected_image in current_image_store:
            img_data = current_image_store[selected_image]
            extra_context = ""
            if doc_context:
                extra_context += "Document context:\n" + doc_context
            if web_context:
                extra_context += ("\n\n" if extra_context else "") + f"Live web results:\n{web_context}"
            prompt_text = f"{extra_context}\n\nQuestion: {user_input}" if extra_context else user_input
            full_response, finish = stream_gemini(
                prompt_text, img_data["bytes"], img_data["mime_type"],
                effective_system_prompt, history, placeholder, acc,
            )
        else:
            if doc_context:
                context = doc_context + (f"\n\nLive web results:\n{web_context}" if web_context else "")
                last_turn = f"Use this context to answer:\n\n{context}\n\nQuestion: {user_input}"
            elif web_context:
                last_turn = (f"Use these live web search results to answer accurately:\n\n"
                             f"{web_context}\n\nQuestion: {user_input}")
            else:
                last_turn = user_input
            messages_to_send = (
                [{"role": "system", "content": effective_system_prompt}]
                + history
                + [{"role": "user", "content": last_turn}]
            )
            full_response, finish = stream_groq(selected_model, messages_to_send, placeholder, acc)
    except Exception as e:
        error_text = friendly_error(e)
        _record_failure(error_text)
        placeholder.error(error_text)
        st.button("Retry", icon=":material/refresh:", key="retry_btn", on_click=retry_failed)
        show_flashes()
        st.stop()
    except BaseException:
        # Streamlit interrupted the run (e.g. the user clicked something mid-stream):
        # keep the partial answer instead of leaving a question with no reply.
        if acc["text"]:
            partial = acc["text"] + "\n\n_(stopped)_"
            current_messages.append({"role": "assistant", "content": partial})
            save_message(st.session_state.user_id, chat_id, "user", user_input)
            save_message(st.session_state.user_id, chat_id, "assistant", partial)
        elif current_messages and current_messages[-1]["role"] == "user":
            current_messages.pop()
        raise

    if not full_response.strip():
        if finish == "length":
            error_text = ("✂️ The model used its whole token budget on reasoning and returned no answer. "
                          "Raise **Max response tokens** in Admin settings, then retry.")
        else:
            error_text = "🤔 The model returned an empty reply. Please retry."
        _record_failure(error_text)
        placeholder.error(error_text)
        st.button("Retry", icon=":material/refresh:", key="retry_btn", on_click=retry_failed)
        show_flashes()
        st.stop()

    if finish == "length":
        notes.append("✂️ Reply cut off at the Max response tokens limit — raise it in Admin settings for longer answers.")

    links = [f"- [{s['title']}]({s['url']})" for s in web_sources if s.get("url")]
    if links:
        full_response += SOURCES_MARKER + "\n".join(links)

    # Persist first, render second, so an interrupted render can't lose the reply.
    # Saved only after a successful reply, so failed requests don't leave orphan questions in the DB.
    current_messages.append({"role": "assistant", "content": full_response, "notes": notes})
    save_message(st.session_state.user_id, chat_id, "user", user_input)
    save_message(st.session_state.user_id, chat_id, "assistant", full_response)
    placeholder.markdown(full_response)
    for note in notes:
        st.caption(note)

# FIX: the sidebar was drawn before this message was processed, so the new chat title,
# the chat order and the "Download Chat" export were one turn out of date. Redraw once.
st.rerun()
