# AI Engineer Take-Home — DPO Fine-Tuning with Soup

**Verdict: DON'T SHIP**

This repository contains my take-home submission for a DPO fine-tuning experiment on a Google Colab Tesla T4 using Soup layer streaming. The final training run completed successfully and produced a non-zero LoRA adapter, but the held-out preference evaluation did not demonstrate a convincing target-task improvement. I therefore chose **DON'T SHIP**.

## Final setup

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`, pinned to revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Hardware: Google Colab Tesla T4, 15,360 MiB VRAM
- Python: 3.12.3
- Soup: 0.75.1
- PyTorch: 2.14.1+cu130
- Transformers: 5.17.0
- TRL: 0.29.1
- PEFT: 0.21.1
- Training: DPO, LoRA r=16 / alpha=32, batch size 2, gradient accumulation 4, one epoch, gradient checkpointing, RAM layer streaming, no quantization

The open-source dataset `d0rj/full-hh-rlhf-ru` is used as a **proxy**, not as real customer-support data. Final data contains 500 training preference pairs and 50 deterministic held-out pairs.

## Key result

The final run completed 63/63 steps with exit code 0. The saved adapter contained 196/196 non-zero LoRA-B tensors and reloaded successfully. However, held-out preference improvement was not convincing:

- baseline mean preference margin: `15.9766`
- tuned mean preference margin: `16.1983`
- mean delta: `+0.2217 nats`
- bootstrap 95% CI: `[-0.4219, +0.8899]`
- fraction improved: `0.54`
- sign-test p-value: `0.336`
- chosen > rejected: `0.46 -> 0.46`

`soup ship` also returned `DON'T SHIP`, but its target-task score was non-discriminative (`0.0000 -> 0.0000`), so I do not treat it as independent evidence of no improvement. Its bundled general checks detected no >5% regression.

The full reasoning, memory analysis, limitations and AI-use disclosure are in [`report.pdf`](report.pdf).

## Repository map

```text
.
├── README.md
├── report.pdf
├── takehome_final.ipynb
├── soup.yaml
├── prepare_data.py
├── materialize_model.py
├── verify_training.py
├── run_final.sh
├── check_config_keys.py
├── token_audit.py
├── extract_notebook_logs.py
├── SHA256SUMS.txt
├── output/
│   └── dpo_qwen15b_stream/
│       ├── adapter_config.json
│       ├── README.md
│       ├── tokenizer_config.json
│       └── tokenizer.json
├── data/
│   ├── train.jsonl
│   ├── eval.jsonl
│   ├── ship_tasks.jsonl
├── artifacts/
│   ├── baseline_logps.json
│   ├── post_logps.json
│   ├── control_zero_adapter.json
│   ├── control_random_adapter.json
│   ├── soup_ship_verdict.json
│   └── soup_ship_evidence.json
└── logs/
    ├── data_rebuild_final.txt
    ├── model_materialization_final.txt
    ├── train_final_v3/
    │   ├── training.log
    │   ├── run_info.txt
    │   ├── nvidia_smi_before.txt
    │   ├── nvidia_smi_after.txt
    │   └── nvidia_smi_sampling.csv
    ├── train_final_v3_aborted_prompt/
    │   └── ...
    ├── verify/
    │   ├── baseline_final_rebuild.log
    │   └── post_final.log
    ├── ship/
    │   └── soup_ship_final.log
    └── notebook_extract/
        ├── INDEX.txt
        ├── KEY_EVIDENCE.txt
        └── cell_*.txt
```

Exact final package versions are recorded in the notebook and final logs.

## Evidence map

Some standalone intermediate files from the first Colab runtime were lost when the VM reset. Their executed cells, outputs, timestamps and hashes remained saved in `takehome_final.ipynb`. To keep this evidence searchable, `extract_notebook_logs.py` copies preserved cell outputs verbatim into `logs/notebook_extract/`. These extracted files are **not presented as the original standalone logs or as reruns**.

Important preserved evidence includes:

| Evidence | Path |
|---|---|
| Initial Python 3.13 install resolving Soup 0.72.4 | `logs/notebook_extract/cell_000.txt` |
| Soup 0.72.4 confirmation | `logs/notebook_extract/cell_003.txt` |
| English / unsuitable official test split check | `logs/notebook_extract/cell_010.txt` (measurement), `cell_012.txt` (manifest/context) |
| PyPI version requirements and initial preflight checks | `logs/notebook_extract/cell_020.txt` |
| Token audit script | `token_audit.py` |
| Prompt-length audit output | `logs/notebook_extract/cell_036.txt` |
| **Pre-training memory estimate and timestamp** | `logs/notebook_extract/cell_037.txt` |
| Raw-text / TRL rendering checks | indexed in `logs/notebook_extract/KEY_EVIDENCE.txt` |
| Verification script frozen before final training | `logs/notebook_extract/cell_056.txt`, `cell_057.txt`, `cell_063.txt` |
| Pre-R9 token-boundary failure evidence | `logs/notebook_extract/cell_063.txt`–`cell_076.txt` |
| Aborted final launch caused only by missing `-y` | `logs/notebook_extract/cell_086.txt` |
| Final launch and input hashes | `logs/notebook_extract/cell_087.txt` |

`logs/notebook_extract/INDEX.txt` maps every extracted file back to the corresponding notebook source cell.

## Memory evidence

The pre-training memory estimate is preserved in the notebook with timestamp **2026-09-30 11:29:32 UTC**, before profiling, dry-run and training. The original estimate predicted roughly **3.7–5.2 GiB** peak GPU memory. Soup later forecast about **10.06 GB**, while raw one-second `nvidia-smi` sampling during the final run peaked at **14,215 MiB (13.882 GiB)**. The report explains the main miss: my original budget treated logits as a single fp16 tensor, while Soup budgets a substantially larger logits/loss-path peak.

## Reproducing the core workflow

The exact pinned model is materialized locally by `materialize_model.py`; data is rebuilt by `prepare_data.py`; training is launched by `run_final.sh`; and held-out verification is performed by `verify_training.py`.

The most important final hashes recorded in the final run are:

```text
soup.yaml          82ea3a7ca287bb51272f667aa1d2cdb84ccd6bdbff96d1bf8e1390da447767e3
data/train.jsonl   175f3ad1ae66034c557f7d54efe30121c6e5b7e99169127539dacb2207e3df67
data/eval.jsonl    ae130e18936df26dbd9db8921f2c131bff394236cb620e7ec2c12ed834c6b4c2
prepare_data.py    981595fe71f26feb321efe2bf9af8cdbc76365141daeb1f52979c85920b38553
verify_training.py f44ce1a3e7f52e280bb1f9add56c15597fe99d46bba008ec8da932a63d89b822
model weights      dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee
```

The final adapter SHA256 was `78069eebd943d6bb433652581d6562698c5693bc32520c4d22169521e5ad8c42`. The large `adapter_model.safetensors` file is intentionally omitted from this LIGHT submission package; its verified SHA256 from the full run artifact is `78069eebd943d6bb433652581d6562698c5693bc32520c4d22169521e5ad8c42`.

## Notes on the Colab reset

The experiment spans two Colab sessions. Session 1 contains the initial environment diagnosis, data revisions, pre-training memory estimate, cache failure and pre-R9 boundary discovery. After the VM reset, Session 2 rebuilt the final environment/data, reproduced the baseline summary metrics, ran the final training, post-training verification and `soup ship`. Historical failures are intentionally preserved rather than removed.

## AI assistance

AI tools were used for debugging unfamiliar Soup/TRL behaviour, code review and verification design. Suggestions were accepted only after executable checks. Some suggestions were corrected after evidence contradicted them, including the initial held-out split choice and an early verification encoder that did not match TRL tokenisation. The report contains the full disclosure.

**Final environment evidence.** Final-session package versions are preserved in `logs/notebook_extract/cell_080.txt` and the final Transformers 5.17.0 pin is confirmed in `logs/notebook_extract/cell_082.txt`.

**Launch-script note.** `run_final.sh` includes `-y`. The immediately preceding launch without `-y` stopped at Soup's confirmation prompt; preserved notebook outputs `cell_086.txt` and `cell_087.txt` document the aborted launch and the final patch.
