import os
import json
import requests

LID_URL = "https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech/resolve/main/lang_id_results.jsonl"


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


def get_lid_prediction_for_key(key: str, cache: dict, cache_path: str, hf_token: str | None):
    """
    Returns prediction string like: "en", "nospeech", or None if not found/failed.
    """
    # cache hit
    if key in cache:
        return cache[key]

    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    try:
        r = requests.get(LID_URL, headers=headers, stream=True)
        if r.status_code != 200:
            cache[key] = None
            save_cache(cache, cache_path)
            return None
    except requests.RequestException:
        cache[key] = None
        save_cache(cache, cache_path)
        return None

    target = f"\"/data/{key}.mp3\""
    try:
        for line in r.iter_lines():
            if not line:
                continue
            s = line.decode("utf-8")
            if target in s:
                obj = json.loads(s)
                prediction = obj.get("prediction")
                cache[key] = prediction
                save_cache(cache, cache_path)
                return prediction
    except Exception:
        cache[key] = None
        save_cache(cache, cache_path)
        return None

    # not found
    cache[key] = None
    save_cache(cache, cache_path)
    return None
