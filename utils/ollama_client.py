"""
Thin wrapper around the local Ollama Python client.

Handles:
  - listing the models you've already pulled with `ollama pull ...`
  - streaming chat completions token-by-token (for the visible reply)
  - a non-streaming chat call (for internal steps the user shouldn't see
    raw output from, e.g. map-reduce summarization or the God Mode
    routing decision)
  - generating embeddings for the RAG pipeline

Written defensively because the `ollama` python package has changed its
return types (dict vs. object) and field names across versions.
"""

import ollama


def get_available_models() -> list[str]:
    """Return the names of every model currently available in local Ollama."""
    try:
        response = ollama.list()
        models = response.get("models", []) if isinstance(response, dict) else response.models
        names = []
        for m in models:
            if isinstance(m, dict):
                name = m.get("model") or m.get("name")
            else:
                name = getattr(m, "model", None) or getattr(m, "name", None)
            if name:
                names.append(name)
        return sorted(names)
    except Exception:
        # Ollama isn't running, or no models are pulled yet.
        return []


def stream_chat(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_predict: int = 1024,
):
    """
    Generator that yields response text chunks as Ollama produces them.
    Used for the visible, streamed reply to the user.
    """
    stream = ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        think=False,  # don't send the "..." system message to the model
        options={
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
        },
    )
    for chunk in stream:
        if isinstance(chunk, dict):
            content = chunk.get("message", {}).get("content", "")
        else:
            content = getattr(chunk.message, "content", "")
        if content:
            yield content


def chat_once(
    model: str,
    messages: list[dict],
    temperature: float = 0.3,
    num_predict: int = 512,
) -> str:
    """
    Non-streaming chat call. Used for internal/background steps:
    map-reduce summarization passes and the God Mode "do I need to
    search the web?" judgment call — neither should stream into the chat.
    """
    response = ollama.chat(
        model=model,
        messages=messages,
        stream=False,
        think=False,  # don't send the "..." system message to the model
        options={"temperature": temperature, "num_predict": num_predict},
    )
    if isinstance(response, dict):
        return response.get("message", {}).get("content", "")
    return getattr(response.message, "content", "")


def embed_text(model: str, text: str) -> list[float]:
    """Return an embedding vector for a piece of text using an Ollama embedding model."""
    result = ollama.embeddings(model=model, prompt=text)
    if isinstance(result, dict):
        return result.get("embedding", [])
    return getattr(result, "embedding", [])
