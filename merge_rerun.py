"""Merge a greedy thinking run with a sampled re-run of its truncated questions, and print CountQA metrics.

Usage: python merge_rerun.py preds_qwen3.5-4b_bf16_think.jsonl preds_qwen3.5-4b_bf16_think_rerun-sampled.jsonl
Writes <greedy>_merged_summary.json next to the greedy file.
"""
import json, os, sys

from eval_countqa import metrics


def load(path):
    return {r["qid"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def summarize(rows):
    bins = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 10**9)]
    out = {"overall": metrics(rows),
           "focused": metrics([r for r in rows if r["focused"]]),
           "cluttered": metrics([r for r in rows if not r["focused"]])}
    out["by_count"] = {f"{a}-{b if b < 10**9 else 'inf'}": metrics([r for r in rows if a <= r["answer"] <= b]) for a, b in bins}
    return out


greedy, rerun = load(sys.argv[1]), load(sys.argv[2])
merged = [rerun.get(q, r) for q, r in sorted(greedy.items())]
summary = {"replaced": len(rerun), "merged": summarize(merged)}
print(json.dumps(summary, indent=2))
json.dump(summary, open(os.path.splitext(sys.argv[1])[0] + "_merged_summary.json", "w"), indent=2)
