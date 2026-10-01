#!/usr/bin/env python
from pathlib import Path
from huggingface_hub import snapshot_download
import hashlib, shutil, os

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REV = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
EXPECTED_WEIGHT_SHA = "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"

files = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]

snap = Path(snapshot_download(
    MODEL,
    revision=REV,
    allow_patterns=files,
))
dst = Path("/content/models/qwen2.5-1.5b-pinned")
if dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True, exist_ok=True)

for name in files:
    shutil.copy2(snap / name, dst / name, follow_symlinks=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

actual = sha256(dst / "model.safetensors")
print("snapshot:", snap)
print("local model:", dst)
print("weight sha256:", actual)
print("expected     :", EXPECTED_WEIGHT_SHA)
assert actual == EXPECTED_WEIGHT_SHA, "Pinned model weight hash mismatch"
print("LOCAL MODEL PASS")
