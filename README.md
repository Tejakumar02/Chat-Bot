# ChatBot — Local ChatGPT-style UI for Ollama

A Streamlit front end for Ollama with model switching, RAG (PDF upload),
free web search, an autonomous "Smart Mode," and password-protected admin
settings. 100% local and free — the only paid component would be your own
hardware running Ollama.

## 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- At least one chat model pulled, e.g.:
  ```bash
  ollama pull qwen3:4b
  ```
- The embedding model, required for RAG (PDF upload + summarization):
  ```bash
  ollama pull nomic-embed-text
  ```

## 2. Install & run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## 3. Setting the admin password

Admin settings (temperature / top-p / max tokens) are locked behind a
password. Default is `admin123` — **change this before sharing the app**:

```bash
export CHATBOT_ADMIN_PASSWORD="your-own-password"
streamlit run app.py
```

## 4. How the features work

| Feature | Behavior |
|---|---|
| **Model switcher** | Dropdown at the top lists whatever's in `ollama list`. Switch anytime, mid-conversation. |
| **New chat / history** | Sidebar. Session-only — all chats reset when you restart the app (no database, by design). |
| **📎 Doc upload** | Inside the chat input box. PDF only. Activates RAG automatically once attached. |
| **🌐 Web search (manual)** | Toggle above the chat input. Only searches when explicitly turned on for that message. Free, via DuckDuckGo (`ddgs` package) — no API key. |
| **God Mode** | Switch at the bottom of the sidebar. When on, the model makes an extra judgment call each turn: "do I need to search the web for this?" — and searches automatically if so, even without the manual toggle. |
| **Summarization** | If your message contains words like "summarize," "tl;dr," "main points," the app skips chunk-similarity retrieval (which would summarize a near-random slice of the doc) and instead feeds the whole document in directly, or map-reduce summarizes it first if it's too long. |
| **Admin settings** | Temperature / top-p / max response tokens. Password-protected, unlocks per session. |

## 5. Known, accepted limitations

- **RAG is session-global, not per-chat.** Upload a PDF in one chat, switch
  to another chat in the sidebar, and that same document is still active
  there. This was a deliberate scope cut for build speed — worth fixing if
  this becomes more than a demo.
- **Smart Mode's self-judgment is inconsistent.** Small/fast local models
  (which is what this whole setup is built around) are known to be
  unreliable at recognizing their own knowledge gaps — they'll sometimes
  answer confidently instead of triggering a search. Test this specifically
  before depending on it; it's not guaranteed to catch every out-of-context
  question.
- **Chat history is not persisted.** Closing or restarting the app loses
  all chats and the uploaded document. This was an explicit choice for
  simplicity (no database).
- **Free web search (`ddgs`)** scrapes DuckDuckGo with no official API — it
  can occasionally rate-limit or break without warning. There's no paid
  fallback configured.
- **CSS targets Streamlit's internal `data-testid` attributes**, which can
  change between Streamlit versions. If the dark sidebar / layout looks off
  after a Streamlit upgrade, that's the first place to check.

## 6. Project structure

```
chatbot_app/
├── app.py                  # main Streamlit app — UI, session state, orchestration
├── utils/
│   ├── ollama_client.py    # list models, streaming chat, non-streaming chat, embeddings
│   ├── rag.py               # PDF extraction, chunking, retrieval, summarization
│   └── websearch.py         # free DuckDuckGo search wrapper
├── requirements.txt
└── README.md
```
# app_online.py is separate file for online it has some similar features like app.py
Live: https://greenyai.streamlit.app/
