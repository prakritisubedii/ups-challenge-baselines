import os
import json
import requests

VAD_URL = "https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech/resolve/main/vad_results.jsonl"

def load_cache(cache_path: str):
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache: dict, cache_path: str):
    folder = os.path.dirname(cache_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f)

def get_vad_segments_for_key(key: str, cache: dict, cache_path: str, hf_token: str | None):
    """
    Returns list like: [{"start": int, "end": int}, ...]
    start/end are audio sample positions at 16kHz.
    """
    # cache hit
    if key in cache:
        return cache[key]

    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    r = requests.get(VAD_URL, headers=headers, stream=True)
    r.raise_for_status()

    target = f"\"{key}\""
    for line in r.iter_lines():
        if not line:
            continue
        s = line.decode("utf-8")
        if target in s:
            obj = json.loads(s)
            segments = obj[key]["timestamps"]
            cache[key] = segments
            save_cache(cache, cache_path)
            return segments

    # not found
    cache[key] = []
    save_cache(cache, cache_path)
    return []
