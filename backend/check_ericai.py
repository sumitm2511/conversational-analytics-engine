"""
check_ericai.py — run this directly: `python check_ericai.py`

Tests EricAI connectivity in complete isolation from FastAPI, DuckDB, or
anything else in the app. Also dumps the FULL raw response object (not just
.content), since some newer models return an empty/None .content while
emitting real output elsewhere in the response (e.g. a separate "reasoning"
field for chain-of-thought/"thinking" models) — .content alone can look
like total failure when it's actually a max_tokens or response-shape issue.

Drop this file in the same folder as llm_client.py before running it.
"""

import time
import sys
import json

print("Importing llm_client...")
t0 = time.time()
from llm_client import _get_client, MODEL_STRUCTURED

print(f"Import OK ({time.time()-t0:.1f}s)")
print(f"Testing model: {MODEL_STRUCTURED}")
print("Sending a trivial request with a GENEROUS max_tokens (200, not 10) so a")
print("reasoning/thinking preamble has room to finish before hitting the cap...")
print("-" * 60)

client = _get_client()
t0 = time.time()
try:
    resp = client.chat.completions.create(
        model=MODEL_STRUCTURED,
        messages=[{"role": "user", "content": "Reply with exactly one word: OK"}],
        max_tokens=200,
        temperature=0.1,
    )
    elapsed = time.time() - t0
    print(f"SUCCESS in {elapsed:.1f}s")
    print()

    msg = resp.choices[0]
    print("finish_reason:", msg.finish_reason)
    print("message.content:", repr(msg.message.content))

    # Some reasoning-style models expose chain-of-thought in an extra field
    # the standard OpenAI SDK type doesn't know about by name — model_extra
    # (pydantic v2) or model_dump() surfaces it even if .content is empty.
    extra = getattr(msg.message, "model_extra", None)
    if extra:
        print("Extra fields on message (e.g. reasoning content):")
        print(json.dumps(extra, indent=2, default=str))

    print()
    print("FULL raw response object:")
    try:
        print(json.dumps(resp.model_dump(), indent=2, default=str))
    except Exception:
        print(resp)

    print()
    if msg.message.content:
        print("EricAI is reachable AND returning real content.")
    elif msg.finish_reason == "length":
        print("DIAGNOSIS: hit max_tokens before producing visible content.")
        print("This model likely needs a MUCH higher max_tokens than the app")
        print("currently uses in llm_client.py's defaults/callers, probably")
        print("because it spends tokens on an internal reasoning/thinking")
        print("phase before the final answer. Check the 'FULL raw response'")
        print("above for any reasoning/thinking field that confirms this.")
    else:
        print("DIAGNOSIS: content is empty but finish_reason is NOT 'length' —")
        print("check the full raw response above for a refusal/safety reason")
        print("or an unexpected response shape from this specific model.")

except Exception as e:
    elapsed = time.time() - t0
    print(f"FAILED after {elapsed:.1f}s")
    print(f"{type(e).__name__}: {e}")
    sys.exit(1)