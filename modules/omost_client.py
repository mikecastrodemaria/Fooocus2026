# Layout / Omost client helpers.
#
# Pure, Gradio-free functions:
#   get_omost_system_prompt() : the Omost system prompt (from the vendored canvas module).
#   call_omost_llm()          : POST to an OpenAI-compatible chat/completions endpoint.
#   parse_canvas()            : run the LLM code in the Omost Canvas namespace, return a dict.
#   flatten_to_prompt()       : turn the layout dict into a readable, deduplicated SDXL prompt.
#
# We do not reinvent a parser: the LLM code runs through Canvas.from_bot_response,
# exactly like the upstream Omost repo.

import requests

from modules.omost_lib.canvas import Canvas, system_prompt as _OMOST_SYSTEM_PROMPT


def get_omost_system_prompt() -> str:
    """Return the Omost system prompt that teaches the LLM the Canvas DSL."""
    return _OMOST_SYSTEM_PROMPT


def call_omost_llm(idea: str, endpoint: str, model: str, timeout: int) -> str:
    """Send the idea to an OpenAI-compatible endpoint and return the raw Canvas code.

    Raises RuntimeError with a clear, human-readable message on any network,
    timeout or protocol error, so the UI can surface it without crashing.
    """
    idea = (idea or "").strip()
    if not idea:
        raise ValueError("The idea is empty: type a phrase to turn into a layout.")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": get_omost_system_prompt()},
            {"role": "user", "content": idea},
        ],
        "temperature": 0.6,
        "stream": False,
    }

    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Timed out after %ss contacting Omost (%s).\n"
            "The model '%s' may take a while to load on the first call: retry once it is "
            "in memory, or raise omost.timeout in config.txt." % (timeout, endpoint, model)
        )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            "Cannot reach Omost (%s): %s\n"
            "Ollama is not responding on this port. Install it from https://ollama.com, start it "
            "(check with the command: ollama list), or fix omost.endpoint in config.txt." % (endpoint, exc)
        )

    if resp.status_code == 404 or 'not found' in (resp.text or '').lower():
        raise RuntimeError(
            "Omost model not found in Ollama (HTTP %s): the model '%s' does not exist.\n"
            "Create it with these commands:\n"
            "  ollama pull hf.co/zhaijunxiao/omost-llama-3-8b-Q8_0-GGUF:Q8_0\n"
            "  ollama cp hf.co/zhaijunxiao/omost-llama-3-8b-Q8_0-GGUF:Q8_0 %s\n"
            "Then check with: ollama list, or adjust omost.model in config.txt." % (resp.status_code, model, model)
        )

    if resp.status_code != 200:
        raise RuntimeError(
            "Omost responded HTTP %s : %s" % (resp.status_code, resp.text[:500])
        )

    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Unexpected Omost response, unrecognized format: %s" % exc)

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Omost returned an empty response.")

    return content


def _build_canvas(code: str) -> Canvas:
    """Run the LLM code through the upstream Canvas namespace.

    If the LLM omitted the markdown fence but the code is otherwise valid, we
    wrap it so from_bot_response can still pick it up. No parsing logic is added.
    """
    text = code if "```python" in code else "```python\n" + code.strip() + "\n```"
    return Canvas.from_bot_response(text)


def parse_canvas(code: str) -> dict:
    """Execute the Canvas code and extract a JSON-serializable layout.

    Returns {"global": {...}, "regions": [{...}]} on success, or
    {"error": "..."} if the code is malformed (never raises).
    """
    try:
        canvas = _build_canvas(code)
    except Exception as exc:
        return {"error": "Invalid Canvas code: %s" % exc}

    try:
        global_block = {
            "prefixes": [str(x) for x in getattr(canvas, "prefixes", [])],
            "suffixes": [str(x) for x in getattr(canvas, "suffixes", [])],
        }
        regions = []
        for comp in getattr(canvas, "components", []):
            regions.append({
                "rect": [int(v) for v in comp.get("rect", [])],
                "distance_to_viewer": float(comp.get("distance_to_viewer", 0.0)),
                "prefixes": [str(x) for x in comp.get("prefixes", [])],
                "suffixes": [str(x) for x in comp.get("suffixes", [])],
            })
        return {"global": global_block, "regions": regions}
    except Exception as exc:
        return {"error": "Could not extract the layout: %s" % exc}


def flatten_to_prompt(layout: dict) -> str:
    """Flatten a layout dict into a readable, deduplicated SDXL prompt.

    Global description first, then regions ordered from closest to farthest.
    Duplicate fragments (regions repeat the global prefix) are removed,
    matching case-insensitively while preserving first-seen order.
    """
    if not isinstance(layout, dict) or "global" not in layout:
        return ""

    phrases = []
    global_block = layout.get("global", {})
    phrases.extend(global_block.get("prefixes", []))
    phrases.extend(global_block.get("suffixes", []))

    regions = layout.get("regions", [])
    regions = sorted(regions, key=lambda r: r.get("distance_to_viewer", 0.0))
    for region in regions:
        phrases.extend(region.get("prefixes", []))
        phrases.extend(region.get("suffixes", []))

    seen = set()
    out = []
    for phrase in phrases:
        if not isinstance(phrase, str):
            continue
        clean = phrase.strip().rstrip(".").strip()
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)

    return ", ".join(out)
