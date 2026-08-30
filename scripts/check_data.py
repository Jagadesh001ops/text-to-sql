import json
from pathlib import Path

for split, path in [("train", "data/processed/train.jsonl"), ("dev", "data/processed/dev.jsonl")]:
    lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
    ex = json.loads(lines[0])
    print(f"{split}: {len(lines)} examples")
    print(f"  db_id:    {ex['db_id']}")
    print(f"  question: {ex['question']}")
    print(f"  sql:      {ex['sql']}")
    print(f"  schema snippet: {ex['schema_text'][:60]}...")
    print()
    