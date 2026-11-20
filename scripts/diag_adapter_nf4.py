"""Diagnostic: does a trained adapter behave the same when run UNMERGED on the 4-bit base it was
trained against, as when merged into the bf16 base and served by vLLM?

    python scripts/diag_adapter_nf4.py --adapter training/outputs_lc/n1000/best_checkpoint --limit 200 --tag n1000

Greedy decoding, same prompts and same extract_sql as eval/vllm_eval.py. Reports the empty-prediction
rate and execution accuracy on the first --limit dev questions (compare with the same slice of the
vLLM predictions: python scripts/diag_adapter_nf4.py --compare eval/results/curve/ft1000_predictions.jsonl).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "eval"))

from utils import GENERATION_STOP_STRINGS, build_prompt, extract_sql, load_config  # noqa: E402
from scorer import score_predictions  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--tag", default="diag")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--out-dir", default="eval/results/diag")
    p.add_argument("--compare", default=None, help="predictions.jsonl from vLLM: print the same-slice stats and exit")
    a = p.parse_args()
    cfg = load_config()
    dev = [json.loads(l) for l in open(cfg["data"]["dev_file"]) if l.strip()][: a.limit]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.compare:
        rows = [json.loads(l) for l in open(a.compare)][: a.limit]
        empty = sum(not r["sql_pred"].strip() for r in rows)
        print(f"vLLM slice of first {len(rows)}: empty={empty} ({empty / len(rows):.1%})")
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base_id = cfg["model"]["base_id"]
    tok = AutoTokenizer.from_pretrained(base_id, padding_side="left")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(model, a.adapter).eval()

    prompts = [build_prompt(r["schema_text"], r["question"]) for r in dev]
    raws: list[str] = []
    for i in range(0, len(prompts), a.batch):
        batch = prompts[i:i + a.batch]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=128, do_sample=False, stop_strings=GENERATION_STOP_STRINGS[:4],
                                 tokenizer=tok, pad_token_id=tok.pad_token_id)
        raws += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"{len(raws)}/{len(prompts)}", flush=True)

    pred_path = out_dir / f"{a.tag}_predictions.jsonl"
    with open(pred_path, "w") as f:
        for rec, text in zip(dev, raws):
            f.write(json.dumps({"db_id": rec["db_id"], "question": rec["question"], "sql_pred": extract_sql(text),
                                "sql_gold": rec["sql"], "difficulty": "unknown", "request_error": None}) + "\n")
    agg = score_predictions(pred_path, Path(cfg["data"]["raw_dir"]), timeout=cfg["eval"]["timeout_seconds"],
                            out_path=out_dir / f"{a.tag}_scores.json")
    empty = sum(not extract_sql(t).strip() for t in raws)
    print(f"{a.tag} (NF4 base + unmerged adapter, first {len(dev)}): empty={empty} ({empty / len(dev):.1%}) "
          f"exec_acc={agg['overall']['exec_acc']:.4f}")


if __name__ == "__main__":
    main()
