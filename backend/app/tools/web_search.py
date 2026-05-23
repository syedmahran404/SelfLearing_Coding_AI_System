"""Web search tool — minimal, network-required.

We avoid third-party SDKs. Two backends are supported, selected by the
presence of API keys in the environment:

- ``SEARCH_PROVIDER=duckduckgo`` (default if no key) — uses DuckDuckGo's
  Instant Answer JSON endpoint. Best-effort, free.
- ``SEARCH_PROVIDER=tavily`` with ``TAVILY_API_KEY`` — JSON results
  with snippets.

This tool is only registered when ``SANDBOX_NETWORK_DEFAULT=allow``.
Without that, agents can't request it (registry permission gate refuses).
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from app.tools.base import BaseTool, Permission, ToolInput, ToolResult, ToolSchema


class WebSearchTool(BaseTool):
    schema = ToolSchema(
        name="web_search",
        description="Search the web. Returns a list of {title, url, snippet}.",
        permissions=[Permission.NETWORK],
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "snippet": {"type": "string"},
                        },
                    },
                }
            },
        },
        default_timeout_s=15,
    )

    async def run(self, ti: ToolInput) -> ToolResult:
        query = str(ti.args["query"])
        max_results = int(ti.args.get("max_results", 5))
        provider = os.environ.get("SEARCH_PROVIDER", "duckduckgo").lower()
        try:
            if provider == "tavily":
                results = await self._tavily(query, max_results)
            else:
                results = await self._duckduckgo(query, max_results)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, error=f"search_failed: {exc}")
        return ToolResult(ok=True, output={"results": results})

    async def _duckduckgo(self, query: str, n: int) -> list[dict[str, Any]]:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": "1", "no_redirect": "1"}
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        out: list[dict[str, Any]] = []
        if data.get("AbstractURL"):
            out.append(
                {
                    "title": data.get("Heading") or query,
                    "url": data["AbstractURL"],
                    "snippet": data.get("AbstractText") or "",
                }
            )
        for item in data.get("RelatedTopics") or []:
            if not isinstance(item, dict):
                continue
            text = item.get("Text") or ""
            url2 = item.get("FirstURL") or ""
            if not url2:
                continue
            out.append({"title": text[:80], "url": url2, "snippet": text})
            if len(out) >= n:
                break
        return out[:n]

    async def _tavily(self, query: str, n: int) -> list[dict[str, Any]]:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY not set")
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": api_key, "query": query, "max_results": n},
            )
            r.raise_for_status()
            data = r.json()
        return [
            {
                "title": x.get("title") or "",
                "url": x.get("url") or "",
                "snippet": x.get("content") or "",
            }
            for x in data.get("results") or []
        ]
