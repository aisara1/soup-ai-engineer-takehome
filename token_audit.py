import json
from pathlib import Path
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_PROMPT = 512
MAX_TOTAL = 1024

tok = AutoTokenizer.from_pretrained(MODEL)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def ids(text):
    return tok(text, add_special_tokens=False)["input_ids"]

def audit(rows, name):
    n = len(rows)

    prompt_over = 0
    chosen_total_over = 0
    rejected_total_over = 0
    either_total_over = 0
    asymmetric_total_over = 0

    prompt_lens = []
    chosen_total_lens = []
    rejected_total_lens = []

    for r in rows:
        p = ids(r["prompt"])
        c = ids(r["chosen"])
        rej = ids(r["rejected"])

        # DPO completions receive EOS if not already present
        if tok.eos_token_id is not None:
            if not c or c[-1] != tok.eos_token_id:
                c = c + [tok.eos_token_id]
            if not rej or rej[-1] != tok.eos_token_id:
                rej = rej + [tok.eos_token_id]

        lp = len(p)
        lc = lp + len(c)
        lr = lp + len(rej)

        prompt_lens.append(lp)
        chosen_total_lens.append(lc)
        rejected_total_lens.append(lr)

        p_over = lp > MAX_PROMPT
        c_over = lc > MAX_TOTAL
        r_over = lr > MAX_TOTAL

        prompt_over += p_over
        chosen_total_over += c_over
        rejected_total_over += r_over
        either_total_over += (c_over or r_over)
        asymmetric_total_over += (c_over != r_over)

    def pct(x):
        return round(100 * x / n, 2)

    def q(xs, p):
        xs = sorted(xs)
        idx = min(len(xs)-1, round((len(xs)-1)*p))
        return xs[idx]

    print(f"\n=== {name} ===")
    print("rows:", n)

    print(
        "prompt tokens p50/p90/p99/max:",
        q(prompt_lens, .50),
        q(prompt_lens, .90),
        q(prompt_lens, .99),
        max(prompt_lens),
    )

    print(
        "chosen total p50/p90/p99/max:",
        q(chosen_total_lens, .50),
        q(chosen_total_lens, .90),
        q(chosen_total_lens, .99),
        max(chosen_total_lens),
    )

    print(
        "rejected total p50/p90/p99/max:",
        q(rejected_total_lens, .50),
        q(rejected_total_lens, .90),
        q(rejected_total_lens, .99),
        max(rejected_total_lens),
    )

    print(f"prompt > {MAX_PROMPT}: {prompt_over}/{n} ({pct(prompt_over)}%)")
    print(f"chosen total > {MAX_TOTAL}: {chosen_total_over}/{n} ({pct(chosen_total_over)}%)")
    print(f"rejected total > {MAX_TOTAL}: {rejected_total_over}/{n} ({pct(rejected_total_over)}%)")
    print(f"either branch > {MAX_TOTAL}: {either_total_over}/{n} ({pct(either_total_over)}%)")
    print(f"asymmetric overflow: {asymmetric_total_over}/{n} ({pct(asymmetric_total_over)}%)")

for split in ["train", "eval"]:
    audit(load_jsonl(f"data/{split}.jsonl"), split)