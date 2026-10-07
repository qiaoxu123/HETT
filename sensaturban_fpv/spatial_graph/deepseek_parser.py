"""Phase 4: map an instruction onto the ontology, using a hosted model.

The division of labour is the round's whole point.  The map graph knows what
entities exist; the geometry teacher knows what relations *mean*; this module's
only job is to say which of those relations the sentence used.  It is never
asked which of a road's segments is meant, and it is never shown anything about
the answer.

**What the model may not see.**  No target id, no target position, no candidate
index, no candidate list, no answer object, and no annotation that is only
reachable through the target.  The prompt carries the instruction text and the
*relation ontology* -- a fixed list of relation names with one-line geometric
glosses -- and nothing else.  That is checkable, so it is checked: a test asserts
that none of those strings can reach the request body.

**Everything is cached and everything is priced.**  A cache keyed on the exact
request means an instruction is never paid for twice, and the cache is on disk so
a re-run costs nothing.  Token usage is recorded per call when the API returns
it, so the cost of a round can be stated rather than estimated.

**The key comes from the environment only.**  Never from the repo, never from a
config file, never written to a log.  :func:`api_key` returns ``None`` when the
variable is unset, and every entry point fails with a message naming the
variable rather than silently degrading -- a parser that quietly returns nothing
would be indistinguishable from a corpus with no relations in it.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# The ontology, as the model sees it: a name and a one-line gloss.  Kept here
# rather than imported from relation_geometry so that the *prompt* is a stable,
# reviewable artefact and cannot drift when the geometry code is refactored.
ONTOLOGY = {
    "north_of": "target is further north (larger world y) than the anchor",
    "south_of": "target is further south (smaller world y) than the anchor",
    "east_of": "target is further east (larger world x) than the anchor",
    "west_of": "target is further west (smaller world x) than the anchor",
    "near": "target is close to the anchor",
    "far": "target is far from the anchor",
    "between": "target lies between the two named anchors",
    "across_road": "the straight line from the anchor to the target crosses a road",
    "same_side_of_road": "anchor and target are on the same side of a road",
    "opposite_side_of_road": "anchor and target are on opposite sides of a road",
    "along_road": "the target lies along the direction of a named road from the anchor",
    "near_intersection": "target is close to where two named roads meet",
    "aligned_with": "the target lies along the anchor's own long axis",
    "parallel_to": "the target's long axis is parallel to the anchor's",
    "perpendicular_to": "the target's long axis is perpendicular to the anchor's",
    "UNKNOWN": "the sentence uses a relation that none of the above describes",
    "ASSOCIATED_WITH": "the anchor is mentioned but no spatial relation is asserted",
}

SYSTEM_PROMPT = (
    "You convert one navigation instruction into JSON. You are given the "
    "sentence and a fixed list of spatial relations. You do not see the map, "
    "the answer, or any candidate. Do not guess which of several places with "
    "the same name is meant: report the name, and let a later stage resolve it.\n"
    "\n"
    "Return a single JSON object with exactly these keys:\n"
    '  "target_phrase": the noun phrase naming the thing to find\n'
    '  "anchors": [{"phrase", "entity_name", "entity_type"}]\n'
    '  "relations": [{"source", "relation_raw", "relation_normalized", '
    '"anchor", "confidence"}]\n'
    "\n"
    "relation_normalized must be one of the relation names given to you, or "
    '"UNKNOWN" when the sentence uses a relation none of them describes, or '
    '"ASSOCIATED_WITH" when the anchor is only mentioned without a spatial '
    "relation. Do not force a mapping. If the relation is not in the list, say "
    "UNKNOWN. confidence is your own 0-1 estimate.\n"
    "\n"
    "Answer with JSON only. No prose, no code fences."
)


DEFAULT_BASE_URL = "https://api.deepseek.com"


def api_key(env=None) -> str | None:
    """The key, from the environment and nowhere else.

    ``ANTHROPIC_AUTH_TOKEN`` is accepted as well because that is the name this
    machine's DeepSeek credential is stored under -- the same key serves both
    the Anthropic-compatible and the OpenAI-compatible endpoints.  It is still
    read from the environment and never from a file.
    """
    env = env if env is not None else os.environ
    return (env.get("DEEPSEEK_API_KEY")
            or env.get("ANTHROPIC_AUTH_TOKEN") or None)


def base_url(env=None) -> str:
    env = env if env is not None else os.environ
    return env.get("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL


def call_model(request: dict, key: str, cache: "Cache" = None,
               timeout: float = 90.0, base: str = None) -> tuple:
    """One completion, served from the cache when it has been asked before.

    Returns ``(parsed_json_or_None, usage)``.  The key is never logged, never
    stored in the cache, and never included in the request body -- it goes in
    the Authorization header and nowhere else.
    """
    import urllib.request

    if cache is not None:
        hit = cache.get(request)
        if hit is not None:
            return hit, {}

    body = json.dumps(request).encode("utf-8")
    url = f"{(base or base_url()).rstrip('/')}/chat/completions"
    http = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(http, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    usage = payload.get("usage") or {}
    content = payload["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = None
    if cache is not None:
        cache.put(request, parsed, usage)
    return parsed, usage


def require_key(env=None) -> str:
    key = api_key(env)
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Phase 4 needs it; it is deliberately "
            "not read from the repository, and no result is produced without "
            "it. Export it and re-run.")
    return key


def build_request(instruction: str, ontology=None, model="deepseek-chat",
                  temperature=0.0, shuffle_seed: int = 0):
    """The exact request body.  Everything the model is allowed to see.

    Assembled here, in one function, so that the leakage test has a single place
    to inspect -- a request built in several places is one that can only be
    spot-checked.
    """
    ontology = ontology or ONTOLOGY
    items = list(ontology.items())
    if shuffle_seed:
        import random
        random.Random(shuffle_seed).shuffle(items)
    listing = "\n".join(f"- {name}: {gloss}" for name, gloss in items)
    user = (f"Instruction:\n{instruction}\n\n"
            f"Available relations:\n{listing}")
    return {
        "model": model,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    }


# Strings that must never appear in a request.  The list is explicit rather
# than a pattern so that adding a field to the round forces a decision here.
#
# Only identifier-shaped tokens, deliberately.  An earlier version included the
# English word "answer", which the system prompt contains for a legitimate
# reason -- it tells the model it is not being shown one -- so the guard fired
# on its own instructions.  A guard that trips on prose teaches the reader to
# ignore it.
FORBIDDEN_IN_REQUEST = (
    "target_id", "target_position", "target_positions", "is_target",
    "candidate_id", "candidate_ids", "candidate_index", "gt_",
    "object_ids", "ann_ids", "description_landmarks", "referenced_landmarks",
)


def check_no_leakage(request: dict) -> None:
    """Raise if anything in the request could reveal the answer."""
    blob = json.dumps(request).lower()
    for token in FORBIDDEN_IN_REQUEST:
        if token.lower() in blob:
            raise ValueError(f"request contains {token!r}")


class Cache:
    """On-disk request cache, so an instruction is never paid for twice."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def key(self, request: dict) -> str:
        blob = json.dumps(request, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def get(self, request: dict):
        path = self.root / f"{self.key(request)}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text())["response"]
        self.misses += 1
        return None

    def put(self, request: dict, response: dict, usage: dict = None) -> None:
        if usage:
            self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        path = self.root / f"{self.key(request)}.json"
        path.write_text(json.dumps({"request": request, "response": response,
                                    "usage": usage or {}}, indent=1) + "\n")

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": (self.hits / total) if total else 0.0,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens}


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

REQUIRED_KEYS = ("target_phrase", "anchors", "relations")


def validate(parsed) -> tuple:
    """(ok, reason).  A model answer is data, and is treated as such."""
    if not isinstance(parsed, dict):
        return False, "not an object"
    for key in REQUIRED_KEYS:
        if key not in parsed:
            return False, f"missing {key}"
    if not isinstance(parsed["anchors"], list) or not isinstance(
            parsed["relations"], list):
        return False, "anchors/relations must be lists"
    for anchor in parsed["anchors"]:
        if not isinstance(anchor, dict) or "entity_name" not in anchor:
            return False, "anchor without entity_name"
    for relation in parsed["relations"]:
        name = relation.get("relation_normalized") if isinstance(relation, dict) \
            else None
        if name not in ONTOLOGY:
            return False, f"relation {name!r} not in ontology"
        confidence = relation.get("confidence")
        if confidence is not None and not (0.0 <= float(confidence) <= 1.0):
            return False, "confidence out of range"
    return True, ""


def normalisation_consistency(first: dict, second: dict) -> dict:
    """How stable the mapping is when the ontology order changes.

    Section 19 asks for the same instruction to be parsed twice with the
    relation list presented in a different order.  A model that is reading the
    sentence will return the same relations; one that is reading position will
    not, and that difference is invisible in a single run.
    """
    def names(parsed):
        return [r.get("relation_normalized") for r in parsed.get("relations", [])
                if isinstance(r, dict)]

    def anchors(parsed):
        return [a.get("entity_name") for a in parsed.get("anchors", [])
                if isinstance(a, dict)]

    a_names, b_names = names(first), names(second)
    a_anchors, b_anchors = anchors(first), anchors(second)
    return {
        "relations_equal": a_names == b_names,
        "relation_jaccard": _jaccard(a_names, b_names),
        "anchors_equal": a_anchors == b_anchors,
        "anchor_jaccard": _jaccard(a_anchors, b_anchors),
    }


def _jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def unknown_rate(parsed_rows) -> dict:
    total = known = unknown = associated = 0
    for parsed in parsed_rows:
        for relation in parsed.get("relations", []) or []:
            if not isinstance(relation, dict):
                continue
            total += 1
            name = relation.get("relation_normalized")
            if name == "UNKNOWN":
                unknown += 1
            elif name == "ASSOCIATED_WITH":
                associated += 1
            elif name in ONTOLOGY:
                known += 1
    return {"relations": total, "mapped": known, "unknown": unknown,
            "associated_with": associated,
            "unknown_rate": unknown / total if total else 0.0,
            "mapped_rate": known / total if total else 0.0}
