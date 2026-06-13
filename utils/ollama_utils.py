"""
ollama_utils.py
Ollama LLM helpers for extracting YOLOE-compatible object descriptors
from natural language user instructions.
"""

import requests


OLLAMA_HOST = 'http://localhost:11434'


def is_ollama_running(host=OLLAMA_HOST):
    """Check if the Ollama server is reachable."""
    try:
        r = requests.get(f'{host}/api/tags', timeout=3)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def gpu_status(host=OLLAMA_HOST):
    """
    Show which Ollama models are currently loaded in VRAM.
    Returns list of dicts with name, size, processor.
    """
    r = requests.get(f'{host}/api/ps', timeout=5)
    r.raise_for_status()
    models = r.json().get('models', [])
    if not models:
        print('[Ollama] No model loaded in VRAM.')
    for m in models:
        print(f"[Ollama] Loaded: {m['name']}  size={m.get('size_vram', '?')} B  "
              f"processor={m.get('details', {}).get('processor', '?')}")
    return models


def unload_model(model, host=OLLAMA_HOST):
    """
    Unload a model from VRAM immediately to free GPU memory.
    Call this after extract_yoloe_prompt() before running YOLOE + Any6D.
    """
    payload = {'model': model, 'keep_alive': 0,
               'prompt': '', 'stream': False}
    r = requests.post(f'{host}/api/generate', json=payload, timeout=10)
    r.raise_for_status()
    print(f'[Ollama] Model "{model}" unloaded from VRAM.')


def list_models(host=OLLAMA_HOST):
    """Return list of pulled Ollama model names."""
    r = requests.get(f'{host}/api/tags', timeout=5)
    r.raise_for_status()
    return [m['name'] for m in r.json().get('models', [])]


def extract_yoloe_prompt(user_input, model='llama3.2:3b', host=OLLAMA_HOST):
    """
    Use Ollama to extract a YOLOE-compatible object name from a
    natural language user instruction.

    Args:
        user_input : str  — e.g. "pick up the yellow mustard bottle on the table"
        model      : str  — Ollama model name (default: llama3.2:3b)
                           use 'pose-extractor' if you ran create_custom_model.sh
        host       : str  — Ollama server URL
    Returns:
        object_name : str — e.g. "mustard bottle"
    Raises:
        RuntimeError if Ollama server is not running
    """
    if not is_ollama_running(host):
        raise RuntimeError(
            'Ollama server is not running.\n'
            'Start it with:  ollama serve\n'
            'Or install it:  bash models/ollama/install_ollama.sh'
        )

    system = (
        'You are an assistant for an open-vocabulary robotic vision system. '
        'Extract only the object name from the user instruction. '
        'Return 1 to 4 words maximum. No explanation, no punctuation.'
    )

    payload = {
        'model'     : model,
        'prompt'    : f'{system}\n\nInstruction: {user_input}\nObject name:',
        'stream'    : False,
        'keep_alive': 0,                   # unload from VRAM immediately after
        'options'   : {'temperature': 0.0} # deterministic output
    }

    r = requests.post(f'{host}/api/generate', json=payload, timeout=30)
    r.raise_for_status()

    raw    = r.json()['response'].strip().strip('"').strip("'").lower()
    result = raw.splitlines()[0].strip()   # keep only first line

    print(f'[Ollama] "{user_input}" → "{result}"')
    return result


def extract_yoloe_prompts(user_input, model='pose-extractor', host=OLLAMA_HOST):
    """
    Use Ollama to extract a list of 2-4 YOLOE-compatible object descriptors
    from a natural language instruction.

    Args:
        user_input : str  — e.g. "pick up the yellow mustard bottle"
        model      : str  — Ollama model (use 'pose-extractor' after rebuild)
        host       : str  — Ollama server URL
    Returns:
        prompts : list of str — e.g. ["mustard bottle", "yellow bottle", ...]
    """
    if not is_ollama_running(host):
        raise RuntimeError(
            'Ollama server is not running.\n'
            'Start it with:  sudo systemctl start ollama'
        )

    payload = {
        'model'     : model,
        'prompt'    : f'Instruction: {user_input}\nOutput:',
        'stream'    : False,
        'keep_alive': 0,
        'options'   : {'temperature': 0.0}
    }

    r = requests.post(f'{host}/api/generate', json=payload, timeout=30)
    r.raise_for_status()
    raw = r.json()['response'].strip()

    # parse the list safely
    try:
        import ast
        # find the list [...] in the response even if there is extra text
        start = raw.find('[')
        end   = raw.rfind(']') + 1
        prompts = ast.literal_eval(raw[start:end])
        prompts = [str(p).strip().lower() for p in prompts if str(p).strip()]
    except Exception:
        # fallback: split by comma if parsing fails
        prompts = [p.strip().strip('"').strip("'").lower()
                   for p in raw.strip('[]').split(',') if p.strip()]

    print(f'[Ollama] "{user_input}"')
    print(f'         → {prompts}')
    return prompts
