"""CountQA eval for a VLM served by llama-server (OpenAI-compatible API).

Follows the CountQA paper protocol (arXiv 2508.06585): paper's system prompt, temperature 0,
Exact Match + RA@5% / RA@10%. First-number regex extraction replaces the paper's Gemini rewriter.
Resumable: predictions are appended to a JSONL file.
"""
import argparse, base64, glob, io, json, math, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from PIL import Image

SYSTEM_PROMPT = (
    "You are a helpful assistant that counts the number of items in an image. The user will provide "
    "an image and ask a question about the number of a certain type of item in the image. If the user "
    "question is referring to multiple objects, it means that you need to provide a sum of the number "
    "of items. You will count the number of items and return the number as an integer. Your output "
    "should STRICTLY be a single integer and nothing else."
)
WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}


def parse_count(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    m = re.search(r"-?\d[\d,]*", text)
    if m:
        return int(m.group().replace(",", ""))
    m = re.search(r"\b(" + "|".join(WORDS) + r")\b", text.lower())
    return WORDS[m.group(1)] if m else None


def encode_image(img_bytes, max_pixels):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        s = math.sqrt(max_pixels / (w * h))
        img = img.resize((max(32, int(w * s)), max(32, int(h * s))), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def load_items(data_dir):
    items = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.parquet"))):
        for row in pd.read_parquet(f).itertuples():
            for qi, (q, a) in enumerate(zip(row.questions, row.answers)):
                items.append(dict(img=row.image["bytes"], question=q, answer=int(a), focused=bool(row.is_focused),
                                  categories=list(row.categories), qid=f"{os.path.basename(f)}:{row.Index}:{qi}"))
    return items


def metrics(rows):
    n = len(rows)
    if n == 0:
        return {}
    ci = lambda p: 1.96 * math.sqrt(p * (1 - p) / n) * 100
    em = sum(r["pred"] == r["answer"] for r in rows) / n
    ra = lambda t: sum(r["pred"] is not None and abs(r["pred"] - r["answer"]) <= t * r["answer"] for r in rows) / n
    errs = [abs(r["pred"] - r["answer"]) for r in rows if r["pred"] is not None]
    return {"n": n, "EM": round(em * 100, 2), "EM_ci95": round(ci(em), 2),
            "RA@5%": round(ra(0.05) * 100, 2), "RA@10%": round(ra(0.10) * 100, 2),
            "MAE": round(sum(errs) / len(errs), 2) if errs else None,
            "unparsed": sum(r["pred"] is None for r in rows),
            "truncated": sum(r.get("finish_reason") == "length" for r in rows),
            "strict_format_rate": round(sum(bool(re.fullmatch(r"\s*\d+\s*", r["raw"])) for r in rows) / n * 100, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--max-pixels", type=int, default=4096 * 32 * 32)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default=None, help="model name for the request (needed by vLLM)")
    ap.add_argument("--workers", type=int, default=1, help="concurrent requests")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--seed", type=int, help="per-request sampling seed (vLLM)")
    ap.add_argument("--rerun-truncated-from", help="only run questions that hit max_tokens in this predictions JSONL")
    args = ap.parse_args()

    items = load_items(args.data)[: args.limit]
    if args.rerun_truncated_from:
        trunc = {r["qid"] for r in map(json.loads, open(args.rerun_truncated_from, encoding="utf-8"))
                 if r.get("finish_reason") == "length"}
        items = [it for it in items if it["qid"] in trunc]
        print(f"re-running {len(items)} truncated questions from {args.rerun_truncated_from}", flush=True)
    done = {}
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            r = json.loads(line)
            done[r["qid"]] = r
    print(f"{len(items)} questions, {len(done)} already done", flush=True)

    t0 = time.time()
    lock = threading.Lock()
    # Ignore http_proxy etc.: clusters often set a proxy whose no_proxy list misses 127.0.0.1.
    session = requests.Session()
    session.trust_env = False
    session.mount("http://", requests.adapters.HTTPAdapter(pool_maxsize=max(10, args.workers)))

    def run_one(it):
        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encode_image(it["img"], args.max_pixels)}},
                    {"type": "text", "text": it["question"]}]},
            ],
            "temperature": args.temperature, "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": args.thinking},
        }
        if args.model:
            payload["model"] = args.model
        for k, v in (("top_p", args.top_p), ("top_k", args.top_k), ("seed", args.seed)):
            if v is not None:
                payload[k] = v
        r = session.post(args.url, json=payload, timeout=3600)
        r.raise_for_status()
        j = r.json()
        ch = j["choices"][0]
        msg = ch["message"]
        raw = msg.get("content") or ""
        rec = {k: v for k, v in it.items() if k != "img"} | {"raw": raw, "pred": parse_count(raw)}
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if reasoning is not None:
            rec["reasoning"] = reasoning
        rec["finish_reason"] = ch.get("finish_reason")
        rec["completion_tokens"] = (j.get("usage") or {}).get("completion_tokens")
        return rec

    todo = [it for it in items if it["qid"] not in done]
    with open(args.out, "a", encoding="utf-8") as fout, ThreadPoolExecutor(max(1, args.workers)) as ex:
        futs = {ex.submit(run_one, it): it for it in todo}
        for fut in as_completed(futs):
            it = futs[fut]
            rec = fut.result()
            with lock:
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                done[it["qid"]] = rec
                k = len(done)
                if k % 25 == 0 or k <= 3:
                    el = time.time() - t0
                    print(f"[{k}/{len(items)}] {el/60:.1f}min  q={it['question']!r} gt={it['answer']} raw={rec['raw']!r}  running={metrics(list(done.values()))['EM']}", flush=True)

    rows = [done[it["qid"]] for it in items]
    summary = {"overall": metrics(rows),
               "focused": metrics([r for r in rows if r["focused"]]),
               "cluttered": metrics([r for r in rows if not r["focused"]])}
    bins = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 10**9)]
    summary["by_count"] = {f"{a}-{b if b < 10**9 else 'inf'}": metrics([r for r in rows if a <= r["answer"] <= b]) for a, b in bins}
    print(json.dumps(summary, indent=2))
    json.dump(summary, open(os.path.splitext(args.out)[0] + "_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
