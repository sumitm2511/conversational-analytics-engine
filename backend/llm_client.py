"""
llm_client.py — Centralized EricAI client wrapper.

Replaces the per-file `Groq(api_key=os.getenv("GROQ_API_KEY"))` pattern used
previously in intent_classifier.py, sql_generator.py, insight_engine.py, and
prediction_engine.py (ClusterEngine + SupervisedClassifierEngine).

EricAI is OpenAI-compatible (same /v1/chat/completions wire format), so this
wrapper uses the standard `openai` SDK pointed at EricAI's base_url, with:
  - SSO token auth via the `ericai` CLI (correct for local/dev laptop use —
    see the PRODUCTION NOTE below for the Service Principal alternative)
  - Retry/backoff tuned to EricAI's documented failure modes (429 short
    backoff, 5xx backoff in MINUTES not seconds, optional cold-start fallback)
  - A JSON-mode wrapper that retries on invalid JSON, since EricAI's
    response_format={"type": "json_object"} is not a hard guarantee the way
    Groq's is.

Requires the `ericai` CLI to be installed and logged in once interactively:
  pip install --extra-index-url https://arm.sero.gic.ericsson.se/artifactory/api/pypi/proj-swtech-pypi-local/simple "ericai[k8s,proxy]"
  ericai --ericsson-access-token   # one-time browser login

No GROQ_API_KEY (or any env var) is required for this local/SSO path — the
token is fetched fresh from the `ericai` CLI on each cache miss.
"""

import os
import json
import time
import threading
import subprocess
from typing import Optional

import httpx
from openai import OpenAI, RateLimitError, APIStatusError

# ── CRITICAL: must be unset, or the openai SDK can silently route traffic to
# OpenAI's public servers instead of EricAI if this env var happens to be set. ──
os.environ.pop("OPENAI_API_KEY", None)


# ── Config ────────────────────────────────────────────────────────────────

# SSO path — correct for a local dev laptop (personal SSO token via the
# `ericai` CLI). This is what you should use right now.
ERICAI_BASE_URL = "https://ray.sero.gic.ericsson.se/backendsso/v1"

# ── PRODUCTION NOTE ──────────────────────────────────────────────────────
# Once this service moves to a real deployment target with an Entra ID
# Service Principal (ERICAI_SP_CLIENT_ID / ERICAI_CLIENT_SECRET / TENANT_ID
# from an IT Support "SSO Application Registration" request), switch to:
#   ERICAI_BASE_URL = "https://ray.sero.gic.ericsson.se/backendsp/v1"
# and replace _TokenCache._fetch() below with the client_credentials OAuth2
# flow against login.microsoftonline.com. Mixing the two up (SSO token
# against /backendsp, or SP token against /backendsso) returns a 401
# immediately, so it's self-diagnosing if you ever get this wrong.
# If deploying inside a GIC K8s cluster instead, use the functional-user +
# CronJob pattern: read the token from a mounted file instead —
#   def _fetch(self) -> str:
#       token_file = os.getenv("ERICAI_TOKEN_FILE", "/app/ericai-token/ERICAI_API_KEY")
#       with open(token_file) as f:
#           return f.read().strip()
# — and keep ERICAI_BASE_URL on /backendsso (functional-user tokens are SSO
# tokens too). The rest of this file (retry logic, chat_completion,
# chat_completion_json) is unchanged regardless of which auth path is used.

# ── MODEL — TEMPORARILY REVERTED FOR A LIVE AVAILABILITY TEST ─────────────
# Timeline:
#   1. This file originally pointed every role at "Qwen/Qwen2.5-32B-Instruct".
#   2. `ericai eristatus --pinned` showed that model no longer appears in the
#      pinned deployment list AT ALL — confirmed root cause of every
#      "EricAI failed after 3 retries: ...DeploymentUnavailableError...".
#      Every retry was guaranteed to fail identically regardless of backoff.
#   3. Swapped to two models confirmed actually Running in that same
#      eristatus output: "Qwen/Qwen3.8-27B-FP8-1M" (primary) and
#      "openai/gpt-oss-120b" (fallback). That swap fixed the outage, but
#      testing then showed Qwen3.8-27B-FP8-1M is a reasoning/"thinking"
#      model — it spends completion tokens on an internal `reasoning` field
#      before writing the real answer to `content`, which risks silent
#      truncation (empty content) if `max_tokens` is too low for a given
#      prompt.
#   4. Deliberately reverted back to "Qwen/Qwen2.5-32B-Instruct" here, for
#      ALL four roles, ONLY to test whether Ericsson has restored that
#      original deployment. Run check_ericai.py after this change — it
#      imports MODEL_STRUCTURED from this file, so it will automatically
#      test against whatever is set below.
#
#   - If check_ericai.py now succeeds with real content (not a
#     DeploymentUnavailableError): the original model is back, and it's
#     safe to just keep this reverted state — Qwen2.5-32B-Instruct was
#     never observed to be a reasoning model, so the empty-content risk
#     goes away too.
#   - If it fails again with DeploymentUnavailableError: the original model
#     is still gone, and the working fallback values from step 3 above
#     should be restored (Qwen/Qwen3.8-27B-FP8-1M / openai/gpt-oss-120b),
#     ideally with the max_tokens values in chat_completion/
#     chat_completion_json raised to give the reasoning model room to
#     think AND answer.
MODEL_STRUCTURED   = "Qwen/Qwen2.5-32B-Instruct"  # intent classification, label/feature extraction
MODEL_SQL          = "Qwen/Qwen2.5-32B-Instruct"  # sql_generator primary
MODEL_SQL_FALLBACK = "Qwen/Qwen2.5-32B-Instruct"  # sql_generator fallback
MODEL_INSIGHT      = "Qwen/Qwen2.5-32B-Instruct"  # insight_engine narrative generation


# ── Token cache (thread-safe, ~1hr lifetime, 60s safety margin) ────────────

class _TokenCache:
    def __init__(self):
        self._token: Optional[str] = None
        self._expires_at: float = 0
        self._lock = threading.Lock()

    def get(self) -> str:
        with self._lock:
            if self._token and time.time() < self._expires_at - 60:
                return self._token
            self._token = self._fetch()
            self._expires_at = time.time() + 3300  # ~55 min; ericai tokens run ~1hr
            return self._token

    def _fetch(self) -> str:
        # Shells out to the ericai CLI, which handles SSO token acquisition
        # (silently refreshing an existing browser session, or prompting an
        # interactive login the very first time). --fail-on-interactive-login
        # makes this script-safe: it errors out instead of hanging if a
        # browser login would otherwise be required.
        try:
            result = subprocess.run(
                ["ericai", "--ericsson-access-token", "--fail-on-interactive-login"],
                capture_output=True, text=True, check=True, timeout=30,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "The `ericai` CLI was not found on PATH. Install it with:\n"
                '  pip install --extra-index-url https://arm.sero.gic.ericsson.se/artifactory/api/pypi/proj-swtech-pypi-local/simple "ericai[k8s,proxy]"'
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ericai --ericsson-access-token failed: {e.stderr.strip()}\n"
                "Run `ericai --ericsson-access-token` manually once to complete "
                "the interactive browser login, then retry."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("ericai --ericsson-access-token timed out after 30s.")

        token = result.stdout.strip()
        if not token:
            raise RuntimeError("ericai --ericsson-access-token returned an empty token.")
        return token

    def invalidate(self):
        with self._lock:
            self._token = None


_token_cache = _TokenCache()
_client: Optional[OpenAI] = None
_client_lock = threading.Lock()


def _get_client() -> OpenAI:
    global _client
    with _client_lock:
        if _client is None:
            _client = OpenAI(
                base_url=ERICAI_BASE_URL,
                api_key=_token_cache.get(),
                http_client=httpx.Client(
                    timeout=httpx.Timeout(connect=10.0, read=180.0, write=10.0, pool=5.0),
                    # EricAI's own curl examples use `-k` (skip TLS verification) —
                    # its cert chain isn't in the default system trust store, so
                    # httpx's default verification rejects the connection outright
                    # ("Connection error" from the openai SDK). Matching that here.
                    verify=False,
                ),
            )
        return _client


def _refresh_client():
    """Force a fresh token + client after a 401."""
    global _client
    _token_cache.invalidate()
    with _client_lock:
        _client = None


def _is_rate_limit(exc) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError):
        msg = str(exc).lower()
        if "request queue is full" in msg or "queue" in msg:
            return True
    return False


# ── Public API ────────────────────────────────────────────────────────────

def chat_completion(
    messages: list,
    model: str = MODEL_INSIGHT,
    temperature: float = 0.2,
    max_tokens: int = 700,
    json_mode: bool = False,
    max_retries: int = 3,
    fallback_model: Optional[str] = None,
) -> str:
    """
    Drop-in replacement for the old
    `client.chat.completions.create(...).choices[0].message.content` calls.
    Returns the raw content string.

    Retry behavior (per EricAI's documented failure modes):
      - 401            -> refresh token, retry immediately
      - 429 / "queue is full" -> short exponential backoff (1s, 2s, 4s)
      - 5xx            -> backoff in MINUTES (60s, 120s, 180s), since this
                          usually means a cold-start or off-hours scale-down;
                          on the second-to-last attempt, switches to
                          `fallback_model` if one was given.

    Every retry now prints a status line — previously this loop was
    completely silent, which made a legitimate 6-minute worst-case 5xx
    backoff (60s+120s+180s across 3 attempts) look identical to the app
    just being hung, with zero way to tell the difference from the console.
    """
    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_exc = None
    for attempt in range(max_retries):
        print(f"[llm_client] Calling EricAI (model={kwargs['model']}, attempt {attempt + 1}/{max_retries})...")
        t0 = time.time()
        try:
            client = _get_client()
            resp = client.chat.completions.create(**kwargs)
            print(f"[llm_client] EricAI responded in {time.time() - t0:.1f}s")
            return resp.choices[0].message.content
        except APIStatusError as e:
            last_exc = e
            elapsed = time.time() - t0
            if e.status_code == 401:
                print(f"[llm_client] 401 after {elapsed:.1f}s — refreshing token and retrying immediately.")
                _refresh_client()
                continue
            if _is_rate_limit(e):
                wait = 2 ** attempt
                print(f"[llm_client] Rate limited / queue full after {elapsed:.1f}s — retrying in {wait}s "
                      f"(attempt {attempt + 1}/{max_retries}).")
                time.sleep(wait)
                continue
            if e.status_code >= 500:
                wait = 60 * (attempt + 1)
                if fallback_model and attempt == max_retries - 2:
                    print(f"[llm_client] Switching to fallback_model={fallback_model} for the next attempt.")
                    kwargs["model"] = fallback_model
                print(f"[llm_client] EricAI returned {e.status_code} after {elapsed:.1f}s — this usually means a "
                      f"cold start or the model deployment being unavailable. Retrying in {wait}s "
                      f"(attempt {attempt + 1}/{max_retries}). Run `ericai eristatus --pinned` in a terminal "
                      f"to check model availability directly, independent of this app.")
                time.sleep(wait)
                continue
            print(f"[llm_client] Non-retryable error ({e.status_code}) after {elapsed:.1f}s — giving up.")
            raise
        except RateLimitError as e:
            last_exc = e
            wait = 2 ** attempt
            print(f"[llm_client] Rate limited after {time.time() - t0:.1f}s — retrying in {wait}s.")
            time.sleep(wait)
            continue

    print(f"[llm_client] Gave up after {max_retries} attempts — EricAI did not respond successfully.")
    raise RuntimeError(f"EricAI failed after {max_retries} retries: {last_exc}")


def chat_completion_json(
    messages: list,
    model: str = MODEL_STRUCTURED,
    temperature: float = 0.1,
    max_tokens: int = 512,
    max_json_retries: int = 2,
) -> dict:
    """
    Like chat_completion(json_mode=True), but ALSO retries when the model
    returns text that isn't valid JSON — EricAI's structured-output support
    is not a hard guarantee the way Groq's response_format enforcement is.
    Raises json.JSONDecodeError if still invalid after max_json_retries.
    """
    msgs = list(messages)
    raw = None
    for attempt in range(max_json_retries + 1):
        raw = chat_completion(
            msgs, model=model, temperature=temperature,
            max_tokens=max_tokens, json_mode=True,
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt == max_json_retries:
                raise
            print(f"[llm_client] EricAI returned invalid JSON (attempt {attempt + 1}/{max_json_retries + 1}) "
                  f"— re-prompting with a correction instruction.")
            msgs = msgs + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "That was not valid JSON. Reply with valid JSON only, no markdown, no explanation."},
            ]