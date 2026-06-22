"""
Free web search via DuckDuckGo. No API key required, no cost.

Uses the `ddgs` package (the current name of what used to be published as
`duckduckgo_search`). Falls back to the old import path automatically if
someone's environment still has the legacy package installed.
"""

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover - fallback for older environments
    from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Return a list of {title, url, snippet} dicts from DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href") or r.get("url", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
    except Exception as e:
        # Surface the failure as a single "result" so the model can tell the
        # user search failed, instead of silently answering from memory.
        return [{"title": "Web search unavailable", "url": "", "snippet": str(e)}]


def format_search_context(results: list[dict]) -> str:
    """Turn search results into a plain-text block to inject into the prompt."""
    if not results:
        return ""
    lines = ["Web search results (most recent information available):"]
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r['title']}\n{r['url']}\n{r['snippet']}\n")
    return "\n".join(lines)
