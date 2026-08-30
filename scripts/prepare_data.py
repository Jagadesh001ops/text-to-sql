"""Phase 1 — Spider data preprocessing.

Downloads the Spider dataset (if not already present) and writes two JSONL
files: data/processed/train.jsonl and data/processed/dev.jsonl.

Each line is a JSON object:
{
  "db_id":       "concert_singer",
  "question":    "How many singers are there?",
  "sql":         "SELECT count(*) FROM singer",
  "schema_text": "Table: stadium\n  ...\nTable: singer\n  ...",
  "prompt":      "<full prompt fed to the model>"
}

Usage:
    python scripts/prepare_data.py [--config configs/config.yaml]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from utils import build_prompt, get_logger, load_config, schema_to_text

log = get_logger("prepare_data")

# Spider is hosted on Yale's site and HuggingFace
SPIDER_HF_URL = (
    "https://huggingface.co/datasets/spider/resolve/main/data/spider.zip"
)


# ── Download ──────────────────────────────────────────────────────────────────

def download_spider(raw_dir: Path) -> None:
    """Download and unzip Spider into raw_dir/spider/ if not already present."""
    spider_dir = raw_dir / "spider"
    if spider_dir.exists() and any(spider_dir.iterdir()):
        log.info("Spider already downloaded at %s — skipping.", spider_dir)
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "spider.zip"

    log.info("Downloading Spider dataset from HuggingFace …")
    token = os.getenv("HF_TOKEN")  # set HF_TOKEN env var if dataset is gated
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    with requests.get(SPIDER_HF_URL, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="spider.zip"
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

    log.info("Extracting …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    zip_path.unlink()
    log.info("Spider extracted to %s", spider_dir)


# ── Schema parsing ────────────────────────────────────────────────────────────

def load_schemas(spider_dir: Path) -> dict[str, list[dict]]:
    """Parse tables.json and return {db_id: [table_dicts]} usable by schema_to_text."""
    tables_path = spider_dir / "tables.json"
    with open(tables_path) as f:
        raw_tables = json.load(f)

    schemas: dict[str, list[dict]] = {}
    for db in raw_tables:
        db_id = db["db_id"]
        col_names_full = db["column_names_original"]   # [[table_idx, col_name], ...]
        col_types = db["column_types"]
        primary_keys = set(db["primary_keys"])          # indices into col_names_full
        foreign_keys = db["foreign_keys"]               # [[from_idx, to_idx], ...]

        # Build per-table structures
        table_map: dict[int, dict] = {
            i: {"table_name": name, "column_names": [], "column_types": [],
                "primary_keys": [], "foreign_keys": []}
            for i, name in enumerate(db["table_names_original"])
        }

        for col_idx, (tbl_idx, col_name) in enumerate(col_names_full):
            if tbl_idx == -1:  # wildcard column — skip
                continue
            tbl = table_map[tbl_idx]
            tbl["column_names"].append(col_name)
            tbl["column_types"].append(col_types[col_idx])
            if col_idx in primary_keys:
                tbl["primary_keys"].append(col_name)

        # Resolve FK indices → names
        col_lookup = {
            idx: (col_names_full[idx][0], col_names_full[idx][1])
            for idx in range(len(col_names_full))
            if col_names_full[idx][0] != -1
        }
        for from_idx, to_idx in foreign_keys:
            if from_idx not in col_lookup or to_idx not in col_lookup:
                continue
            from_tbl_idx, from_col = col_lookup[from_idx]
            to_tbl_idx, to_col = col_lookup[to_idx]
            to_tbl_name = db["table_names_original"][to_tbl_idx]
            table_map[from_tbl_idx]["foreign_keys"].append(
                {"from_col": from_col, "to_table": to_tbl_name, "to_col": to_col}
            )

        schemas[db_id] = list(table_map.values())
    return schemas


# ── Main processing ───────────────────────────────────────────────────────────

def process_split(
    spider_dir: Path,
    schemas: dict[str, list[dict]],
    split: str,           # "train" or "dev"
    out_path: Path,
    few_shot_examples: list[dict] | None = None,
) -> int:
    """Convert one Spider split into a JSONL file. Returns number of examples."""
    src = spider_dir / f"{split}.json"
    with open(src) as f:
        examples = json.load(f)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w") as fout:
        for ex in tqdm(examples, desc=f"Processing {split}"):
            db_id = ex["db_id"]
            question = ex["question"]
            sql = ex["query"]
            difficulty = ex.get("difficulty", "unknown")

            if db_id not in schemas:
                log.warning("Unknown db_id %s — skipping.", db_id)
                continue

            schema_text = schema_to_text(schemas[db_id])
            prompt = build_prompt(schema_text, question, examples=few_shot_examples)

            record = {
                "db_id": db_id,
                "question": question,
                "sql": sql,
                "difficulty": difficulty,
                "schema_text": schema_text,
                "prompt": prompt,
            }
            fout.write(json.dumps(record) + "\n")
            written += 1

    log.info("Wrote %d examples → %s", written, out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Spider dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--few-shot",
        type=int,
        default=0,
        help="Number of few-shot examples to include in dev prompts (0 = zero-shot)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(cfg["data"]["raw_dir"]).parent   # data/raw
    spider_dir = Path(cfg["data"]["raw_dir"])        # data/raw/spider
    processed_dir = Path(cfg["data"]["processed_dir"])

    # 1. Download
    download_spider(raw_dir)

    # 2. Load schemas
    log.info("Parsing schema …")
    schemas = load_schemas(spider_dir)
    log.info("Loaded %d database schemas.", len(schemas))

    # 3. Build few-shot pool from first N train examples (if requested)
    few_shot_pool: list[dict] | None = None
    if args.few_shot > 0:
        with open(spider_dir / "train_spider.json") as f:
            train_raw = json.load(f)
        few_shot_pool = []
        for ex in train_raw[: args.few_shot]:
            db_id = ex["db_id"]
            if db_id in schemas:
                few_shot_pool.append({
                    "schema_text": schema_to_text(schemas[db_id]),
                    "question": ex["question"],
                    "sql": ex["query"],
                })
        log.info("Using %d few-shot examples.", len(few_shot_pool))

    # 4. Process splits
    # Spider has train_spider.json and train_others.json; combine both for training.
    # We'll create a unified train.json in raw for convenience.
    train_combined = spider_dir / "train_combined.json"
    if not train_combined.exists():
        all_train = []
        for fname in ["train_spider.json", "train_others.json"]:
            p = spider_dir / fname
            if p.exists():
                with open(p) as f:
                    all_train.extend(json.load(f))
        with open(train_combined, "w") as f:
            json.dump(all_train, f)
        log.info("Combined %d training examples.", len(all_train))

    # Temporarily rename so process_split finds "train.json"
    train_alias = spider_dir / "train.json"
    if not train_alias.exists():
        shutil.copy(train_combined, train_alias)

    process_split(spider_dir, schemas, "train", Path(cfg["data"]["train_file"]))
    process_split(
        spider_dir, schemas, "dev", Path(cfg["data"]["dev_file"]),
        few_shot_examples=few_shot_pool,
    )

    log.info("✓ Data preparation complete.")


if __name__ == "__main__":
    main()
