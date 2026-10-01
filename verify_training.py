#!/usr/bin/env python
"""Part 2 verification: held-out DPO preference margins, before vs after training.

ONE scoring path for every mode, reproducing TRL 0.29.1 DPOTrainer preprocessing for
non-conversational rows (verified against TRL source):
  1. raw text, NO chat template (TRL 0.29.1: is_conversational=False, template not applied);
  2. append tok.eos_token to chosen/rejected TEXT if absent;
  3. prompt_ids = tok(prompt); full_ids = tok(prompt + completion_with_eos);
  4. assert full_ids starts with prompt_ids; completion_ids = full_ids[len(prompt_ids):];
  5. logp(completion | prompt) = SUM of token log-probs over completion_ids only (causal shift).
Soup 0.75.1 cap_dpo (prompt 512 / total 1024) is asserted to be a no-op for every row.

Modes:
  --baseline          base model on data/eval.jsonl -> artifacts/baseline_logps.json
                      + negative control 1: zero-initialised LoRA (lora_B=0) -> must be NOT_TRAINED
  --adapter PATH      trained adapter -> artifacts/post_logps.json + verdict vs baseline
                      + negative control 2: norm-matched RANDOM adapter -> must NOT show a preference shift
"""
import argparse, datetime, hashlib, json, math, os
import numpy as np, torch, transformers, tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID, MODEL_REV = "Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
EVAL_PATH = "data/eval.jsonl"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_R, LORA_ALPHA = 16, 32
SOUP_PROMPT_CAP, SOUP_TOTAL_CAP = 512, 1024      # Soup 0.75.1 cap_dpo: max_length//2, max_length
ENCODING = "trl-0.29.1-nonconv: eos appended to text; tok(prompt); tok(prompt+completion); suffix slice"
# ---- Pre-registered thresholds (fixed BEFORE training; do not tune after seeing results) ----
NOISE_ABS_TOL = 1e-2          # nats; per-pair |Δ| below this = numerical noise
N_BOOT, BOOT_SEED = 10000, 0  # bootstrap CI for mean Δ
DEVICE = "cuda"

def sha256(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def load_base():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REV, dtype=torch.float16).to(DEVICE)
    model.eval()
    return tok, model

def ids(tok, text):
    return tok(text)["input_ids"]

def encode(tok, rows):
    """Reproduce TRL 0.29.1 non-conversational preprocessing. Returns (enc, encode_stats)."""
    eos = tok.eos_token
    # Tokenizer-call semantics check: default call vs add_special_tokens=False must agree for this tokenizer.
    probe = rows[0]["prompt"] + rows[0]["chosen"]
    special_tokens_agree = ids(tok, probe) == tok(probe, add_special_tokens=False)["input_ids"]
    assert special_tokens_agree, "tokenizer adds special tokens by default; confirm TRL's exact call"
    enc, boundary_diff = [], 0
    for i, r in enumerate(rows):
        p_ids = ids(tok, r["prompt"])
        comps = []
        for field in ("chosen", "rejected"):
            text = r[field] if r[field].endswith(eos) else r[field] + eos
            full = ids(tok, r["prompt"] + text)
            assert full[:len(p_ids)] == p_ids, f"row {i} {field}: full sequence does not start with prompt ids"
            c_ids = full[len(p_ids):]
            assert len(c_ids) >= 1 and c_ids[-1] == tok.eos_token_id, f"row {i} {field}: completion must end with EOS"
            sep = tok(r[field], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            boundary_diff += int(sep != c_ids)
            comps.append(c_ids)
        c_ids, j_ids = comps
        assert len(p_ids) >= 1
        assert len(p_ids) <= SOUP_PROMPT_CAP, f"row {i}: prompt {len(p_ids)} > {SOUP_PROMPT_CAP} (Soup would truncate)"
        assert len(p_ids) + max(len(c_ids), len(j_ids)) <= SOUP_TOTAL_CAP, f"row {i}: total > {SOUP_TOTAL_CAP} (Soup would truncate)"
        enc.append((p_ids, c_ids, j_ids))
    stats = {"special_tokens_default_equals_false": special_tokens_agree,
             "completions_where_separate_tokenization_differs": boundary_diff,
             "completions_total": 2 * len(rows),
             "max_prompt_tok": max(len(p) for p, _, _ in enc),
             "max_total_tok": max(len(p) + max(len(c), len(j)) for p, c, j in enc),
             "soup_cap_noop": True}
    return enc, stats

@torch.inference_mode()
def completion_logp(model, p_ids, c_ids):
    x = torch.tensor([p_ids + c_ids], device=DEVICE)
    logits = model(input_ids=x, use_cache=False).logits          # [1, P+C, V], fp16
    P, C = len(p_ids), len(c_ids)
    lg = logits[0, P - 1:P + C - 1].float()                      # logits at t predict token t+1
    tgt = x[0, P:P + C]
    lp = torch.log_softmax(lg, dim=-1).gather(-1, tgt[:, None]).squeeze(-1)
    assert lp.shape[0] == C and torch.isfinite(lp).all(), "bad logp slice / non-finite"
    return float(lp.sum()), C

def score(model, enc):
    out = []
    for i, (p, c, j) in enumerate(enc):
        lc, nc = completion_logp(model, p, c)
        lr, nr = completion_logp(model, p, j)
        out.append({"i": i, "prompt_tok": len(p), "chosen_tok": nc, "rejected_tok": nr,
                    "logp_chosen": lc, "logp_rejected": lr, "margin": lc - lr,
                    "margin_per_tok": lc / nc - lr / nr})
    return out

def summarize(rows):
    m = np.array([r["margin"] for r in rows])
    return {"n": len(rows),
            "mean_logp_chosen": float(np.mean([r["logp_chosen"] for r in rows])),
            "mean_logp_rejected": float(np.mean([r["logp_rejected"] for r in rows])),
            "mean_margin": float(m.mean()), "median_margin": float(np.median(m)),
            "frac_chosen_gt_rejected": float((m > 0).mean()),
            "mean_margin_per_tok": float(np.mean([r["margin_per_tok"] for r in rows]))}

def compare(base_rows, new_rows):
    """Δ_i = new margin - base margin = (implicit DPO reward margin)/beta, since ref = base (lora_B=0 init)."""
    d = np.array([n["margin"] - b["margin"] for b, n in zip(base_rows, new_rows)])
    rng = np.random.default_rng(BOOT_SEED)
    boots = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    nz = d[np.abs(d) > NOISE_ABS_TOL]; k = int((nz > 0).sum()); n = len(nz)
    p_sign = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0   # one-sided exact
    len_diff = np.array([b["chosen_tok"] - b["rejected_tok"] for b in base_rows], dtype=float)
    corr = float(np.corrcoef(d, len_diff)[0, 1]) if d.std() > 0 and len_diff.std() > 0 else None
    return {"mean_delta": float(d.mean()), "ci95_mean_delta": [float(lo), float(hi)],
            "frac_delta_pos": float((d > NOISE_ABS_TOL).mean()), "frac_delta_neg": float((d < -NOISE_ABS_TOL).mean()),
            "max_abs_delta": float(np.abs(d).max()), "sign_test_p_one_sided": p_sign,
            "corr_delta_vs_len_diff": corr, "deltas": d.tolist()}

def behaviour_shift(cmp):
    return cmp["ci95_mean_delta"][0] > 0 and cmp["mean_delta"] > NOISE_ABS_TOL

def lora_tensors(model):
    return {n: p.detach().float() for n, p in model.named_parameters() if "lora_" in n}

def classify(param_ok, cmp):
    if not param_ok: return "NOT_TRAINED (adapter params absent/zero/not loaded)"
    if cmp["max_abs_delta"] <= NOISE_ABS_TOL: return "NOT_TRAINED (params non-zero but no measurable effect)"
    if behaviour_shift(cmp): return "TRAINED (params changed AND held-out preference margin increased)"
    return "PARAMS_CHANGED_ONLY (outputs changed, no held-out preference shift)"

def meta(tok, enc_stats, extra=None):
    m = {"timestamp_utc": now(), "model_id": MODEL_ID, "model_revision": MODEL_REV,
         "eval_path": EVAL_PATH, "eval_sha256": sha256(EVAL_PATH),
         "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
         "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "dtype": "float16 (log_softmax fp32)",
         "encoding": ENCODING, "encode_stats": enc_stats, "eos_token": tok.eos_token,
         "score": "sum logp over completion tokens",
         "thresholds": {"NOISE_ABS_TOL": NOISE_ABS_TOL, "N_BOOT": N_BOOT, "BOOT_SEED": BOOT_SEED}}
    m.update(extra or {}); return m

def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), ensure_ascii=False, indent=2); print("wrote", path)

def print_summary(name, s): print(f"[{name}] " + json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}))

def run_baseline(args):
    from peft import LoraConfig, get_peft_model
    tok, model = load_base()
    rows = [json.loads(l) for l in open(EVAL_PATH)]
    enc, est = encode(tok, rows)
    print("[encode] " + json.dumps(est))
    base = score(model, enc)
    rep = score(model, enc[:5])                                   # determinism / noise check
    rep_diff = max(max(abs(a["logp_chosen"] - b["logp_chosen"]), abs(a["logp_rejected"] - b["logp_rejected"])) for a, b in zip(base, rep))
    if rep_diff > NOISE_ABS_TOL: print(f"WARNING: repeat diff {rep_diff:.3e} > NOISE_ABS_TOL")
    s = summarize(base); print_summary("baseline", s); print(f"[baseline] repeat_max_abs_diff={rep_diff:.3e}")
    dump({"meta": meta(tok, est, {"mode": "baseline", "repeat_max_abs_diff": rep_diff}),
          "summary": s, "rows": base}, "artifacts/baseline_logps.json")
    # Negative control 1: zero-initialised LoRA through the same PEFT path.
    pm = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS,
                                          lora_dropout=0.0, init_lora_weights=True)).eval()
    lt = lora_tensors(pm); nB = [n for n in lt if "lora_B" in n]
    nB_nonzero = sum(bool(lt[n].abs().sum() > 0) for n in nB)
    ctrl = score(pm, enc); cmp = compare(base, ctrl)
    param_ok = len(nB) > 0 and nB_nonzero == len(nB)
    verdict = classify(param_ok, cmp)
    print(f"[control_zero_adapter] lora_B tensors={len(nB)} nonzero={nB_nonzero} mean_delta={cmp['mean_delta']:.3e} "
          f"max_abs_delta={cmp['max_abs_delta']:.3e} -> {verdict}")
    print("[control_zero_adapter] EXPECTED: NOT_TRAINED ->", "PASS" if verdict.startswith("NOT_TRAINED") else "FAIL")
    dump({"meta": meta(tok, est, {"mode": "control_zero_adapter"}),
          "lora_B_total": len(nB), "lora_B_nonzero": nB_nonzero, "comparison": cmp, "verdict": verdict,
          "rows": ctrl}, "artifacts/control_zero_adapter.json")

def run_post(args):
    from peft import PeftModel
    from safetensors.torch import load_file
    base_art = json.load(open("artifacts/baseline_logps.json"))
    tok, model = load_base()
    assert base_art["meta"]["eval_sha256"] == sha256(EVAL_PATH), "eval file changed since baseline!"
    assert base_art["meta"]["encoding"] == ENCODING, "encoding path differs from baseline!"
    rows = [json.loads(l) for l in open(EVAL_PATH)]; enc, est = encode(tok, rows)
    # --- parameter evidence straight from the saved file ---
    sd = load_file(os.path.join(args.adapter, "adapter_model.safetensors"))
    keys = list(sd); inner = [k for k in keys if ".inner." in k]
    fB = {k: v.float() for k, v in sd.items() if "lora_B" in k}
    fB_nonzero = sum(bool(v.abs().sum() > 0) for v in fB.values())
    pm = PeftModel.from_pretrained(model, args.adapter).eval()
    live = lora_tensors(pm)
    def live_name(k): return k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
    matched = [k for k in keys if live_name(k) in live]
    equal = [k for k in matched if torch.allclose(live[live_name(k)].cpu(), sd[k].float(), atol=1e-3, rtol=1e-3)]
    norms = {k: float(v.norm()) for k, v in fB.items()}
    param_ok = len(fB) > 0 and fB_nonzero == len(fB) and len(equal) == len(keys) and not inner
    post = score(pm, enc); cmp = compare(base_art["rows"], post); verdict = classify(param_ok, cmp)
    s = summarize(post); print_summary("post", s)
    print(f"[params] file_keys={len(keys)} inner_keys={len(inner)} lora_B={len(fB)} lora_B_nonzero={fB_nonzero} "
          f"loaded_and_equal={len(equal)}/{len(keys)} lora_B_norm[min/median/max]="
          f"{min(norms.values()):.3e}/{float(np.median(list(norms.values()))):.3e}/{max(norms.values()):.3e}")
    print(f"[post vs baseline] mean_delta={cmp['mean_delta']:.4f} ci95={cmp['ci95_mean_delta']} "
          f"frac_pos={cmp['frac_delta_pos']:.2f} frac_neg={cmp['frac_delta_neg']:.2f} sign_p={cmp['sign_test_p_one_sided']:.3g} "
          f"corr(Δ,len_diff)={cmp['corr_delta_vs_len_diff']} -> {verdict}")
    dump({"meta": meta(tok, est, {"mode": "post", "adapter": args.adapter}),
          "params": {"file_keys": len(keys), "inner_keys": inner[:5], "lora_B": len(fB), "lora_B_nonzero": fB_nonzero,
                     "loaded_and_equal": len(equal), "lora_B_norms": norms},
          "summary": s, "comparison": cmp, "verdict": verdict, "rows": post}, "artifacts/post_logps.json")
    # Negative control 2: same per-tensor norms, random direction -> params "changed" but should show NO preference shift.
    g = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                r = torch.randn(p.shape, generator=g)
                p.copy_((r / r.norm() * p.detach().float().norm().cpu()).to(p.dtype).to(p.device))
    ctrl = score(pm, enc); ccmp = compare(base_art["rows"], ctrl); cverdict = classify(True, ccmp)
    print(f"[control_random_adapter] mean_delta={ccmp['mean_delta']:.4f} ci95={ccmp['ci95_mean_delta']} -> {cverdict}")
    print("[control_random_adapter] EXPECTED: not TRAINED ->", "PASS" if not cverdict.startswith("TRAINED") else "FAIL")
    dump({"meta": meta(tok, est, {"mode": "control_random_adapter", "seed": 0}),
          "comparison": ccmp, "verdict": cverdict, "rows": ctrl}, "artifacts/control_random_adapter.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--baseline", action="store_true"); g.add_argument("--adapter", type=str)
    a = ap.parse_args()
    run_baseline(a) if a.baseline else run_post(a)
