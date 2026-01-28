# Scripts

This folder contains helper scripts for data inspection and dataset preparation. The most important one for v1 data curation is `build_manifest_vad_clean.py`, which streams the VAD metadata and produces a clean manifest of clips that are strongly speech-dominant, have long contiguous speech segments, and yield multiple trainable 10-second chunks. The script is streaming and does not load the full JSONL into memory.

Recommended v1 settings for clean speech (start here and iterate):
- `--min_speech_ratio 0.30 --min_segment_sec 10 --min_trainable_sec 30 --target_clips 20000 --max_per_tar 200`

Example:
```bash
UPS_ARTIFACT_DIR=/content/drive/MyDrive/ups_artifacts \
  python scripts/build_manifest_vad_clean.py --target_clips 20000
```
