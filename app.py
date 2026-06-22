#/set nothink
"""/bye
ChatBot — a local, ChatGPT-style front end for Ollama.

Features:
  - Model switcher (lists whatever you've pulled in Ollama)
  - New chat + chat history sidebar (session-only, resets on restart)
  - Web search toggle (free, via DuckDuckGo) — only searches when turned on
  - SmartMode — autonomous web-search decision (separate switch, bottom-left)
  - Doc upload (PDF) — activates RAG retrieval; handles summarization
    requests correctly (full-document context, not random chunk similarity)
  - Password-protected admin settings (temperature / top-p / max tokens)
  - Streaming responses, current date & time injected into every turn
  - Defensive error handling around every external call (Ollama, embeddings,
    web search, PDF parsing) so one failure doesn't crash the whole turn

Run with:
    streamlit run app.py

Set a custom admin password with:
    export CHATBOT_ADMIN_PASSWORD="your-password-here"
(defaults to "admin123" if not set — change this before sharing the app)
"""

import os
import uuid
from datetime import datetime

import numpy as np
import streamlit as st

from utils.ollama_client import chat_once, get_available_models, stream_chat
from utils.rag import is_summarization_request, process_uploaded_pdf, retrieve, summarize_document
from utils.websearch import format_search_context, web_search

ADMIN_PASSWORD = os.environ.get("CHATBOT_ADMIN_PASSWORD", "admin123")

st.set_page_config(
    page_title="ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling — dark sidebar / light main area, ChatGPT-style layout
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {background: transparent;}

        # .stApp { background-color: #ffffff; color: #111111; }
        .stApp { background-color: #171717; color: #ececec; }

        .block-container {
            max-width: 820px;
            padding-top: 1.5rem;
            padding-bottom: 7rem;
        }

        section[data-testid="stSidebar"] {
            background-color: #171717;
        }
        section[data-testid="stSidebar"] * {
            color: #ececec !important;
        }
        section[data-testid="stSidebar"] .stButton button {
            background-color: transparent;
            border: 1px solid #3a3a3a;
            border-radius: 8px;
            text-align: left;
            padding: 0.4rem 0.7rem;
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            background-color: #2a2a2a;
            border-color: #2a2a2a;
        }
        section[data-testid="stSidebar"] [data-testid="stExpander"] {
            border: 1px solid #3a3a3a;
            border-radius: 8px;
        }

        section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover,
        section[data-testid="stSidebar"] [data-testid="stExpander"] summary:focus,
        section[data-testid="stSidebar"] [data-testid="stExpander"] details > summary {
            background-color: #171717 !important;
            color: #ececec !important;
        }

        section[data-testid="stSidebar"] [data-testid="stExpander"] .stButton button:hover {
            background-color: transparent !important;
            border-color: #3a3a3a !important;
        }

        section[data-testid="stSidebar"] [data-testid="stExpander"] div[data-baseweb="input"],
        section[data-testid="stSidebar"] [data-testid="stExpander"] div[data-baseweb="input"] input {
            background-color: #171717 !important;
            color: #ececec !important;
            -webkit-text-fill-color: #ececec !important;
        }

        section[data-testid="stSidebar"] [data-testid="stExpander"] div[data-baseweb="input"] input::placeholder {
            color: #8a8a8a !important;
        }
        section[data-testid="stSidebar"] [data-testid="stExpander"] div[data-baseweb="input"] button {
            background-color: #171717 !important;
        }

        section[data-testid="stSidebar"] [data-testid="stExpander"] div[data-baseweb="input"] button svg {
            fill: #ececec !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stToggle"] {
            background-color: rgba(255, 255, 255, 0.08) !important;
            border-radius: 20px !important;
            padding: 4px 8px !important;
        }

        section[data-testid="stSidebar"] [data-testid="stExpander"] div[data-baseweb="input"]:hover,
        section[data-testid="stSidebar"] [data-testid="stExpander"] div[data-baseweb="input"]:focus-within {
            background-color: #171717 !important;
            border-color: #3a3a3a !important;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] .st-cf {
            background-color: rgba(255, 255, 255, 0.25) !important;
        }
        
        

        .sidebar-title {
            font-size: 1.15rem;
            font-weight: 700;
            padding: 0.3rem 0 1rem 0;
        }
        .sidebar-divider { border-color: #2f2f2f; margin: 0.6rem 0; }

        .mode-switch-wrap {
            margin-top: 1.2rem;
            padding-top: 0.8rem;
            border-top: 1px solid #2f2f2f;
        }

        # .empty-state {
        #     text-align: center;
        #     margin-top: 16vh;
        #     color: #6e6e80;
        # }
        # .empty-state h2 { color: #353740; font-weight: 600; }

        .empty-state {
            text-align: center;
            margin-top: 16vh;
            color: #9a9aa5;
        }
        .empty-state h2 { color: #ececec; font-weight: 600; }

        [data-testid="stChatMessage"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        }

        # .doc-badge {
        #     font-size: 0.85rem;
        #     color: #1a7f37;
        #     padding-top: 0.4rem;
        # }
        # .smart-mode-badge {
        #     font-size: 0.8rem;
        #     color: #b8860b;
        #     padding-bottom: 0.3rem;
        # }

        .doc-badge {
            font-size: 0.85rem;
            color: #4ade80;
            padding-top: 0.4rem;
        }
        .smart-mode-badge {
            font-size: 0.8rem;
            color: #fbbf24;
            padding-bottom: 0.3rem;
        }
    
        # }

        # .st-key-websearch_bar {
        #     position: fixed;
        #     bottom: 72px;
        #     left: 50%;
        #     transform: translateX(250px);
        #     z-index: 999;
        #     width: fit-content !important;
        #     zoom: 1.0;
        #     background: #f7f7f8;
        #     padding: 2px 10px;
        #     border-radius: 16px;
        #     box-shadow: 0 -2px 6px rgba(0,0,0,0.06);
        #     font-size: 0.85rem;
        }

        .st-key-websearch_bar {
            position: fixed;
            bottom: 72px;
            left: 50%;
            transform: translateX(250px);
            z-index: 999;
            width: fit-content !important;
            zoom: 1.0;
            background: #2a2a2a;
            border: 1px solid #3a3a3a;
            padding: 2px 10px;
            border-radius: 16px;
            box-shadow: 0 -2px 8px rgba(0,0,0,0.4);
            font-size: 0.85rem;
        }
        .st-key-websearch_bar > div,
        .st-key-websearch_bar [data-testid="stVerticalBlock"],
        .st-key-websearch_bar [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-websearch_bar [data-testid="stElementContainer"],
        .st-key-websearch_bar .element-container {
            background: transparent !important;
            padding: 0 !important;
            margin: 0 !important;
            border: none !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            width: fit-content !important;
        }
        .st-key-model_switcher {
            margin-left: -250px;
            max-width: 140px;
        }
        
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def new_chat() -> None:
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New chat",
        "messages": [],
        "created": datetime.now(),
    }
    st.session_state.current_chat_id = chat_id


def delete_chat(chat_id: str) -> None:
    st.session_state.chats.pop(chat_id, None)
    if st.session_state.current_chat_id == chat_id:
        if st.session_state.chats:
            most_recent = max(st.session_state.chats.items(), key=lambda kv: kv[1]["created"])
            st.session_state.current_chat_id = most_recent[0]
        else:
            new_chat()


def init_state() -> None:
    defaults = {
        "chats": {},
        "current_chat_id": None,
        "model": None,
        "temperature": 0.7,
        "top_p": 0.9,
        "num_predict": 1024,
        "web_search_on": False,
        "smart_mode": False,
        "admin_unlocked": False,
        "rag_chunks": [],
        "rag_embeddings": None,
        "rag_full_text": "",
        "rag_doc_name": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.current_chat_id is None:
        new_chat()


init_state()


def needs_web_search(model: str, user_text: str) -> bool:
    """
    SmartMode judgment call: decide whether this message likely needs
    information outside the model's own knowledge / the active document.
    Non-streaming, low temperature, short — this should never be shown
    to the user directly.

    Known limitation: small models are inconsistent at self-assessing their
    own knowledge gaps. This can under-trigger (answers confidently instead
    of searching) more often than it over-triggers. Test before relying on
    it for a demo.
    """
    judge_messages = [
        {
            "role": "system",
            "content": (
                "Decide if answering the user's message requires current, "
                "real-world, or up-to-date information you might not "
                "reliably know on your own (e.g. recent events, prices, "
                "schedules, current people in roles, news). "
                "Reply with exactly one word: YES or NO."
            ),
        },
        {"role": "user", "content": user_text},
    ]
    try:
        reply = chat_once(model, judge_messages, temperature=0.0, num_predict=5)
        return "yes" in reply.strip().lower()
    except Exception:
        # If the judgment call itself fails, don't block the main response —
        # just fall back to "no auto-search" for this turn.
        return False


# --------------------------------------------------------------------------
# Sidebar — new chat, history, admin settings, mode switch
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">🤖 ChatBot</div>', unsafe_allow_html=True)

    if st.button("➕  New chat", use_container_width=True, key="new_chat_btn"):
        new_chat()
        st.rerun()

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    st.caption("Chats (this session)")

    chat_items = sorted(
        st.session_state.chats.items(), key=lambda kv: kv[1]["created"], reverse=True
    )
    for chat_id, chat in chat_items:
        is_active = chat_id == st.session_state.current_chat_id
        label = ("● " if is_active else "") + chat["title"]
        col_select, col_delete = st.columns([5, 1])
        with col_select:
            if st.button(label, key=f"select_{chat_id}", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        with col_delete:
            if st.button("🗑", key=f"del_{chat_id}"):
                delete_chat(chat_id)
                st.rerun()

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    # --- password-protected admin settings ---------------------------------
    with st.expander("⚙️ Admin settings"):
        if not st.session_state.admin_unlocked:
            st.caption("Locked. Enter the admin password to change these.")
            pw = st.text_input("Password", type="password", key="admin_pw_input")
            if st.button("Unlock", key="admin_unlock_btn"):
                if pw == ADMIN_PASSWORD:
                    st.session_state.admin_unlocked = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
        else:
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
            if st.session_state.rag_doc_name and st.button("Clear uploaded document"):
                st.session_state.rag_chunks = []
                st.session_state.rag_embeddings = None
                st.session_state.rag_full_text = ""
                st.session_state.rag_doc_name = None
                st.rerun()
            if st.button("🔒 Lock settings"):
                st.session_state.admin_unlocked = False
                st.rerun()

    # --- mode switch, deliberately last so it renders at the bottom --------
    st.markdown('<div class="mode-switch-wrap">', unsafe_allow_html=True)
    mode_label = "🔴 Smart-Mode" if st.session_state.smart_mode else "⚪ Normal Mode"
    st.session_state.smart_mode = st.toggle(mode_label, value=st.session_state.smart_mode, key="smart_mode_toggle")
    if st.session_state.smart_mode:
        st.caption("Auto-decides per message if a web search is needed.")
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Model availability check
# --------------------------------------------------------------------------
models = get_available_models()
if not models:
    st.error(
        "No Ollama models found. Make sure Ollama is running and you've pulled "
        "at least one model, e.g.:\n\n`ollama pull qwen3:4b`"
    )
    st.stop()

if st.session_state.model not in models:
    st.session_state.model = models[0]

with st.container(key="model_switcher"):
    st.session_state.model = st.selectbox(
        "Model", models, index=models.index(st.session_state.model), label_visibility="collapsed"
    )

# --------------------------------------------------------------------------
# Message history for the active chat
# --------------------------------------------------------------------------
current_chat = st.session_state.chats[st.session_state.current_chat_id]

if not current_chat["messages"]:
    st.markdown('<div class="empty-state"><h2>What can I help with?</h2></div>', unsafe_allow_html=True)
else:
    for msg in current_chat["messages"]:
        avatar = "🤖" if msg["role"] == "assistant" else "🧑"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

# --------------------------------------------------------------------------
# Tool row — web search toggle + active-document indicator
# --------------------------------------------------------------------------
# tool_col1, tool_col2, _ = st.columns([1.4, 3, 4])
# with tool_col1:
#     st.session_state.web_search_on = st.toggle(
#         "🌐 Web search", value=st.session_state.web_search_on
#     )
# with tool_col2:
#     if st.session_state.rag_doc_name:
#         st.markdown(
#             f'<div class="doc-badge">📄 {st.session_state.rag_doc_name} — RAG active</div>',
#             unsafe_allow_html=True,
#         )
with st.container(key="websearch_bar"):
    st.session_state.web_search_on = st.toggle(
        "🌐 Web search", value=st.session_state.web_search_on
    )
if st.session_state.rag_doc_name:
    st.markdown(
        f'<div class="doc-badge">📄 {st.session_state.rag_doc_name} — RAG active</div>',
        unsafe_allow_html=True,
    )
    
# --------------------------------------------------------------------------
# Chat input — supports attaching a PDF directly (📎 icon inside the box)
# --------------------------------------------------------------------------
prompt = st.chat_input(
    "Message ChatBot...",
    accept_file=True,
    file_type=["pdf"],
)

user_text = ""
uploaded_files = []

if prompt:
    if hasattr(prompt, "text"):
        user_text = prompt.text or ""
        uploaded_files = prompt.files or []
    else:
        # Older Streamlit versions without accept_file support return a plain string.
        user_text = prompt or ""

# Reliability fix: a file attached with no typed text must still be processed,
# not silently dropped because user_text is empty.
if user_text or uploaded_files:
    if not user_text and uploaded_files:
        names_preview = ", ".join(f.name for f in uploaded_files)
        user_text = f"[Uploaded: {names_preview}] Please review this document."

    # --- handle any newly attached PDF(s) first --------------------------
    if uploaded_files:
        with st.spinner("Reading document(s)..."):
            try:
                all_chunks, all_embeds, full_texts, names = [], [], [], []
                for f in uploaded_files:
                    chunks, embeds, full_text = process_uploaded_pdf(f.read())
                    if chunks:
                        all_chunks.extend(chunks)
                        all_embeds.append(embeds)
                        full_texts.append(full_text)
                    names.append(f.name)
                if all_chunks:
                    st.session_state.rag_chunks = all_chunks
                    st.session_state.rag_embeddings = np.vstack(all_embeds)
                    st.session_state.rag_full_text = "\n\n".join(full_texts)
                    st.session_state.rag_doc_name = ", ".join(names)
                else:
                    st.warning("Couldn't extract any text from that PDF (it may be scanned/image-only).")
            except Exception as e:
                st.error(f"⚠️ Couldn't process the uploaded file: {e}")

    # --- save + show the user's message -----------------------------------
    current_chat["messages"].append({"role": "user", "content": user_text})
    if current_chat["title"] == "New chat":
        current_chat["title"] = (user_text[:40] + "…") if len(user_text) > 40 else user_text

    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_text)

    # --- build extra context only from whatever's actually active ---------
    context_blocks = []

    if st.session_state.rag_chunks:
        try:
            if is_summarization_request(user_text):
                with st.spinner("Reading through the document..."):
                    doc_context = summarize_document(
                        st.session_state.model, st.session_state.rag_full_text, chat_once
                    )
                context_blocks.append(f"Full document content/summary for context:\n{doc_context}")
            else:
                with st.spinner("Searching document..."):
                    retrieved = retrieve(
                        user_text, st.session_state.rag_chunks, st.session_state.rag_embeddings
                    )
                if retrieved:
                    doc_ctx = "\n\n".join(f"(page {c['page']}) {c['text']}" for c in retrieved)
                    context_blocks.append(f"Context from uploaded document:\n{doc_ctx}")
        except Exception as e:
            st.warning(
                f"⚠️ Couldn't use the uploaded document this turn ({e}). "
                "Make sure you've run `ollama pull nomic-embed-text`."
            )

    # SmartMode: only run the judgment call if the user hasn't already
    # manually forced search on for this turn.
    auto_search_triggered = False
    if st.session_state.smart_mode and not st.session_state.web_search_on:
        with st.spinner("Deciding whether this needs a web search..."):
            auto_search_triggered = needs_web_search(st.session_state.model, user_text)
        if auto_search_triggered:
            st.markdown(
                '<div class="smart-mode-badge">🔴 Smart-Mode decided this needs a web search.</div>',
                unsafe_allow_html=True,
            )

    run_web_search = st.session_state.web_search_on or auto_search_triggered
    if run_web_search:
        try:
            with st.spinner("Searching the web..."):
                results = web_search(user_text)
            web_ctx = format_search_context(results)
            if web_ctx:
                context_blocks.append(web_ctx)
        except Exception as e:
            st.warning(f"⚠️ Web search failed this turn ({e}). Answering without it.")

    now_str = datetime.now().strftime("%A, %B %d, %Y — %I:%M %p")
    system_prompt = (
        "You are ChatBot, a helpful, concise AI assistant running locally via Ollama.\n"
        f"Current date and time: {now_str}.\n"
        "Only mention the date/time if the user's question actually depends on it."
    )
    if context_blocks:
        system_prompt += (
            "\n\nUse the following retrieved context to answer the user's question. "
            "Prioritize it over your own background knowledge, since it may be more current "
            "or specific. If the context doesn't answer the question, say so instead of guessing.\n\n"
            + "\n\n---\n\n".join(context_blocks)
        )

    messages_for_model = [{"role": "system", "content": system_prompt}] + current_chat["messages"]

    # --- stream the response, with a clean fallback on failure -------------
    try:
        with st.chat_message("assistant", avatar="🤖"):
            response_text = st.write_stream(
                stream_chat(
                    st.session_state.model,
                    messages_for_model,
                    temperature=st.session_state.temperature,
                    top_p=st.session_state.top_p,
                    num_predict=st.session_state.num_predict,
                )
            )
        current_chat["messages"].append({"role": "assistant", "content": response_text})
    except Exception as e:
        st.error(f"⚠️ Couldn't reach Ollama or the model failed: {e}")
        current_chat["messages"].append({
            "role": "assistant",
            "content": (
                "⚠️ Sorry, I couldn't generate a response. Please check that Ollama "
                "is running and the selected model is available, then try again."
            ),
        })

    st.rerun()
