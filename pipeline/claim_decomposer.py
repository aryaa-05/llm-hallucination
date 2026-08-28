"""
Stage A: Claim Decomposer
Exact logic from QA_Consistency_Check notebook.
Provider routing:
    Gemini     (gemini-2.5-flash-lite)  Primary     default
    Groq       (openai/gpt-oss-20b)     Fallback    Gemini 429 / quota / failure
    OpenRouter (openrouter/free)        Emergency   Gemini + Groq failure
"""

import re
import time
import os
import requests
import google.generativeai as genai
from groq import Groq

# ── API setup ────────────────────────────────────────────────
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

GEMINI_MODEL    = "gemini-2.5-flash-lite"
GROQ_MODEL      = "openai/gpt-oss-20b"
OPENROUTER_MODEL = "openrouter/free"

genai.configure(api_key=GEMINI_API_KEY)
_gemini = genai.GenerativeModel(GEMINI_MODEL)
_groq   = Groq(api_key=GROQ_API_KEY)

_provider_state = {
    "current"               : "gemini",
    "gemini_failures"       : 0,
    "groq_failures"         : 0,
    "gemini_cooldown_until" : 0,
}

MIN_CALL_INTERVAL = 2.0
_last_call_time   = {"t": 0.0}


def _call_gemini(prompt: str, temperature: float = 0.0) -> str:
    response = _gemini.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=512,
        )
    )
    return response.text.strip()


def _call_groq(prompt: str, temperature: float = 0.0) -> str:
    response = _groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def _call_openrouter(prompt: str, temperature: float = 0.0) -> str:
    """Emergency provider: OpenAI-compatible REST API via OpenRouter."""
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type" : "application/json",
        },
        json={
            "model"       : OPENROUTER_MODEL,
            "messages"    : [{"role": "user", "content": prompt}],
            "temperature" : temperature,
            "max_tokens"  : 512,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _smart_call_llm(prompt: str, temperature: float = 0.0) -> str:
    """Provider router:
    Gemini (primary) -> Groq (fallback on Gemini failure) -> OpenRouter (emergency).
    """
    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if missing:
        raise RuntimeError(
            "Missing API keys: " + ", ".join(missing)
            + ". Add them under the Space's Settings -> Variables and secrets."
        )
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it under the Space's "
            "Settings -> Variables and secrets (required for the emergency provider)."
        )

    now = time.time()
    use_gemini_first = (
        _provider_state["current"] == "gemini" and
        now >= _provider_state["gemini_cooldown_until"]
    )
    # Gemini primary first, Groq as fallback; OpenRouter handled separately below.
    providers = ["gemini", "groq"] if use_gemini_first else ["groq", "gemini"]
    providers = [p for p in providers if p != "gemini" or now >= _provider_state["gemini_cooldown_until"]]

    last_errs = {}
    for provider in providers:
        try:
            if provider == "groq":
                result = _call_groq(prompt, temperature)
                _provider_state["groq_failures"] = 0
                _provider_state["current"]       = "groq"
                return result
            elif provider == "gemini":
                result = _call_gemini(prompt, temperature)
                _provider_state["gemini_failures"] = 0
                _provider_state["current"]         = "gemini"
                return result
        except Exception as e:
            err = str(e)
            last_errs[provider] = err
            if provider == "gemini" and ("429" in err or "quota" in err.lower()):
                wait_match = re.search(r"retry in (\d+\.?\d*)s", err)
                wait_secs  = float(wait_match.group(1)) + 2 if wait_match else 60
                _provider_state["gemini_failures"]      += 1
                _provider_state["gemini_cooldown_until"] = time.time() + wait_secs
                _provider_state["current"]               = "groq"
            elif provider == "gemini":
                # Any non-429 Gemini failure -> treat as failure, go to Groq
                _provider_state["gemini_failures"] += 1
                _provider_state["current"]          = "groq"
            elif provider == "groq":
                _provider_state["groq_failures"] += 1
            continue

    # Emergency: both Gemini and Groq failed/unavailable -> OpenRouter
    try:
        result = _call_openrouter(prompt, temperature)
        _provider_state["current"] = "openrouter"
        return result
    except Exception as e:
        last_errs["openrouter"] = str(e)

    detail = "; ".join(f"{k}: {v}" for k, v in last_errs.items())
    raise RuntimeError(
        "All providers failed (Gemini, Groq, OpenRouter). Check API keys and quotas.\n"
        f"  GEMINI_API_KEY set: {bool(GEMINI_API_KEY)}"
        f" | GROQ_API_KEY set: {bool(GROQ_API_KEY)}"
        f" | OPENROUTER_API_KEY set: {bool(OPENROUTER_API_KEY)}\n"
        f"  Last errors -> {detail or 'none captured'}"
    )


def call_llm(prompt: str, temperature: float = 0.0) -> str:
    """Rate-limited wrapper around the provider switcher."""
    elapsed = time.time() - _last_call_time["t"]
    if elapsed < MIN_CALL_INTERVAL:
        time.sleep(MIN_CALL_INTERVAL - elapsed)
    _last_call_time["t"] = time.time()
    return _smart_call_llm(prompt, temperature)


# ── Decomposer ───────────────────────────────────────────────
DECOMPOSE_PROMPT = """\
You are an expert fact-checker. Break down the following text into atomic, verifiable factual claims.

Rules:
1. Each claim must be a single, standalone sentence.
2. RESOLVE ALL PRONOUNS AND REFERENCES EXPLICITLY. A claim MUST NOT contain unresolved pronouns like 'She', 'He', 'They', 'It', or 'This'. Replace them with the actual subject from the text.
3. Do NOT correct factual errors. If the text says "The sky is green", extract "The sky is green".
4. Only extract verifiable facts (skip opinions or filler).
5. Output ONLY the most distinct, independently verifiable claims — no redundant or overlapping claims.
6. Output a MAXIMUM of {max_claims} claims total, one per line, with no bullet points or numbering.

Text: {text}
Claims:"""


def decompose_claims(text: str, max_claims: int = 6) -> list:
    prompt = DECOMPOSE_PROMPT.format(text=text, max_claims=max_claims)
    raw    = call_llm(prompt)
    claims = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        # Remove numbers if the LLM adds them anyway
        line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        # Remove bullet points
        line = re.sub(r"^[-*\u2022]\s*", "", line).strip()
        if len(line) > 10:
            claims.append(line)
        if len(claims) >= max_claims:
            break   # Hard cap: never process more than max_claims
    return claims
