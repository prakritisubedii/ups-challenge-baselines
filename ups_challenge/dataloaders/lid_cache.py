# ups_challenge/dataloaders/lid_cache.py

import os
import json
import requests
from typing import Optional

LID_URL = "https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech/resolve/main/lang_id_results.jsonl"


def _normalize_key(key: str) -> str:
    """
    WebDataset keys sometimes include extensions.
    LID metadata uses /data/<key>.mp3, where <key> has no extension.
    """
    base = os.path.basename(key)
    if "." in base:
        base = base.split(".")[0]
    return base


def load_cache(cache_path: str) -> dict:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def save_cache(cache: dict, cache_path: str) -> None:
    folder = os.path.dirname(cache_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp_path, cache_path)


def get_lid_prediction_for_key(
    key: str,
    cache: dict,
    cache_path: str,
    hf_token: Optional[str],
    autosave_every: int = 200,
) -> Optional[str]:
    """
    Returns a string prediction like: "en", "es", "nospeech", ...
    We match against filepath containing: /data/<key>.mp3

    Caching behavior:
      - If key is in cache -> return immediately
      - If not -> stream-scan the jsonl, cache result
      - Save to disk only occasionally (autosave_every) to reduce I/O
    """
    key = _normalize_key(key)

    if key in cache:
        return cache[key]

    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    target_fp = f"/data/{key}.mp3"

    # Stream the LID file line-by-line from HF (no full download into memory)
    r = requests.get(LID_URL, headers=headers, stream=True, timeout=60)
    r.raise_for_status()

    pred: Optional[str] = None

    for raw_line in r.iter_lines():
        if not raw_line:
            continue
        try:
            s = raw_line.decode("utf-8")
        except Exception:
            continue

        if target_fp not in s:
            continue

        # Found the line for this key
        try:
            obj = json.loads(s)
        except Exception:
            obj = None

        if isinstance(obj, dict):
            # Most likely field name is "prediction"
            p = obj.get("prediction", None)

            # Be a bit robust if the field name differs
            if p is None:
                p = obj.get("lang", None) or obj.get("language", None)

            if isinstance(p, str):
                pred = p.lower().strip()
            else:
                pred = None

        break

    # Cache result (even None, so we don't re-scan next time)
    cache[key] = pred

    # Save occasionally so Colab crashes don’t lose progress
    if len(cache) % autosave_every == 0:
        save_cache(cache, cache_path)

    return pred
