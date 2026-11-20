# Two follow-up experiments: why does fine-tuning beat prompting?

**Written before any result exists** (see the git history of this file for the order of events).

The main results show 3-shot prompting at 60.8% and fine-tuning at 71.4%, but that comparison
changes two things at once (the number of examples *and* the mechanism: in-context vs. weight
updates), and the 3-shot prompt *lost* ground on extra-hard queries (35.7% → 31.0%). The three
fixed examples were single-table queries (one easy, two medium), so we guessed that example
*diversity* matters. These experiments test that guess instead of asserting it.

All numbers are Spider dev (1,034 questions), execution accuracy, greedy decoding, bf16, scored
by the same harness as the main results. Both experiments are served through vLLM on the 8 GB
laptop GPU (the same engine for every cell, so cells are comparable with each other; vLLM and the
Hugging Face runs differ by ≤0.3 points on identical models).

## Experiment 1: how many examples in the prompt, and which ones?

**Design.** Base Qwen2.5-3B-Instruct, same prompt format as the main results, with
n ∈ {1, 3, 6} examples chosen by

- `random`: uniform from the training pool, 3 seeds;
- `diverse`: greedy coverage of SQL constructs, difficulty tiers and databases (see `eval/shots.py`),
  3 seeds (the seed changes the candidate subset and tie-breaks);
- plus `first-3` (the original fixed 3-shot run, re-run on this engine) and `0-shot`.

The shot pool is training examples with schemas ≤ 1,500 characters (85% of them), so six-shot
prompts fit the context window. The same pool serves both strategies.

**Hypotheses**

- **H1.1** Accuracy rises from 0 to 3 shots, then flattens (6 shots add < 1.5 points over 3).
- **H1.2** At matched n, `diverse` beats `random` on the hard + extra tiers by ≥ 2 points
  (mean over seeds), and shows no difference on easy.
- **H1.3** `random-3` reproduces the extra-hard drop seen with the original 3 shots (extra-hard
  below the 0-shot value), and `diverse-3` reduces it.

**What would count against us.** If `diverse` and `random` differ by less than the seed-to-seed
spread, the diversity explanation of the extra-hard drop is not supported and the README says so.
With 3 seeds per cell this experiment can only detect fairly large effects; differences smaller
than the spread are reported as "not distinguishable", not as findings.

## Experiment 2: how much training data does fine-tuning need?

**Design.** Same QLoRA recipe as the main run (r=16, lr 2e-4, effective batch 16, 3 epochs,
cosine schedule), trained on random subsets of the 7,000 training examples of size
N ∈ {1,000, 3,000}; the 7,000 point is the existing run. Checkpoint chosen by the lowest dev loss
among evaluations during training, as in the main run. Each model is merged into the bf16 base and
evaluated zero-shot through vLLM.

**Hypotheses**

- **H2.1** Most of the gain comes early. Writing gain(N) = (acc(N) − acc(0-shot)) / (acc(7,000) −
  acc(0-shot)): gain(1,000) ≥ 0.6 and gain(3,000) ≥ 0.85.
- **H2.2** The hard and extra tiers gain proportionally more from additional data than easy does.

**What would count against us.** gain(1,000) < 0.4 means data volume is a bigger lever than we
think; gain(3,000) within one point of the 7,000 model means volume is not the limit and the gain
is mostly format/convention learning.

## Addendum (2026-11-19, after the first two runs, before the rest)

The plan above says vLLM and Hugging Face agree to within 0.3 points on identical models. That
was measured on the fine-tuned model only (71.1% vs 71.4%). On the **base** model the first two
runs differ by more: 0-shot 55.2% (vLLM) vs 55.9% (Hugging Face), and the original 3-shot 59.9%
vs 60.8%, i.e. −0.7 and −1.0 points. The original 3-shot's extra-hard drop reproduces on vLLM
(32.1% → 30.4%). The hypotheses are unchanged; all comparisons within this experiment are
between cells run on the same engine, and cross-engine differences of about a point are treated
as engine noise when relating these results to the main table.

## Caveats, stated up front

- One training run per size (no seeds), so size differences under about 1–1.5 points are noise.
- The dev set is used for checkpoint selection (by loss) and for reporting; same caveat as the
  main results.
- A laptop GPU with a small KV cache: throughput numbers are not part of these experiments, only
  accuracy.
