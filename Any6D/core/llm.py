"""
LLM utilities: send instructions to Ollama and parse the keyword response.

Design rationale
----------------
The pipeline uses a local LLM (Mistral via Ollama) rather than a rule-based
keyword extractor because natural language instructions vary wildly:
  "hand me the thing next to the blue bottle" — requires world knowledge
  "grab the cylindrical steel can"            — needs visual grounding

The LLM is guided by a CALIBRATED_SYSTEM prompt (defined per dataset) that
contains 1000+ instruction→keyword examples so it learns to output the
2-3 word visual descriptors that YOLOE responds to best.

The caller is responsible for choosing the correct system prompt for the
dataset being evaluated. The prompts are NOT shared across datasets because
each dataset has different object appearances and YOLOE detection behaviour.
"""
import re
import requests
from .constants import OLLAMA_URL, LLM_MODEL


def call_llm(instruction: str, system_prompt: str,
             llm_model: str = LLM_MODEL) -> str:
    """
    Send a user instruction to Ollama and return the raw LLM response.

    The system_prompt is the CALIBRATED_SYSTEM string from the pipeline script.
    It teaches the LLM to produce YOLOE-compatible keywords through few-shot
    examples (e.g. "Pick up the steel can" → "tin can").

    Args:
        instruction:   Free-form natural language user request.
        system_prompt: Dataset-specific few-shot calibration prompt.
        llm_model:     Ollama model name (default: mistral:latest).

    Returns:
        Raw LLM text response, or "" if the server is unreachable.
        Always call parse_llm() on this output before using it.

    Note:
        Exceptions are silently caught and return "". The pipeline will then
        fall back to using the last word of the instruction as the keyword.
    """
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": llm_model, "stream": False,
                  "system": system_prompt, "prompt": instruction},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception:
        return ""


def parse_llm(raw: str) -> str:
    """
    Extract a clean 1-3 word keyword from any LLM response format.

    LLMs do not always follow the "output ONLY the keyword" instruction.
    This parser handles the common deviation patterns observed with Mistral:

      1. Quoted text:   'The keyword is "yellow driller"'  → "yellow driller"
      2. Arrow prefix:  "→ rubber duck"                    → "rubber duck"
      3. Label prefix:  "keyword: benchvise"               → "benchvise"
      4. Plain word:    "driller"                          → "driller"
      5. Multiline:     takes the first line with 1-4 words

    Lines with more than 4 words are skipped because they are explanations,
    not keywords. The 4-word ceiling (not 3) gives a small buffer for edge
    cases like "extra large orange clamp".

    Args:
        raw: Unprocessed string from call_llm().

    Returns:
        Cleaned lowercase keyword (1-3 words ideally), or the first 40
        characters of the first line if no clean match is found.
    """
    raw = raw.strip()

    # Priority 1: quoted text is the most explicit signal ("yellow duck")
    m = re.search(r'["""]([^"""]{1,40})["""]', raw)
    if m:
        return m.group(1).strip().lower()

    # Priority 2: arrow pattern matches our calibrated examples format
    m = re.search(r'→\s*(.+)$', raw)
    if m:
        return m.group(1).strip().lower()[:40]

    # Priority 3: strip common verbose prefixes the LLM adds despite instructions
    raw = re.sub(
        r'^(keyword[:\s]+|output[:\s]+|the (object|keyword|word) (is|:)\s*)',
        '', raw, flags=re.I)

    # Priority 4: scan lines for the first one that looks like a keyword
    for line in raw.split('\n'):
        line = re.sub(r'[^\w\s\'-]', '', line.strip().lstrip('-→•*:')).strip()
        if 0 < len(line.split()) <= 4:
            return line.lower()

    # Fallback: truncate whatever we have
    return raw.split('\n')[0][:40].strip().lower()
