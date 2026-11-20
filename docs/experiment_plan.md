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

## Addendum 2 (2026-11-19, after the n ≤ 6 results; extension requested by the project owner)

The n ≤ 6 sweep is complete (see the README). It supports the claim that random examples plateau
by about 3, but only up to 6; whether 8 or 16 examples change that was not tested. This extension
adds n ∈ {8, 16} for both strategies, 3 seeds each (12 runs). It is **post-hoc in scope** (decided
after seeing the first results) and is reported as such; the predictions below are fixed before any
of these runs exist. Worst-case prompts are about 4,500 tokens at n = 16, inside the 6,144-token
context.

- **H1.4** The random-example plateau persists: mean accuracy at n = 8 and n = 16 is within 1.5
  points of the n = 6 mean.
- **H1.5** Diverse selection converges toward random as n grows (coverage saturates): the absolute
  diverse − random gap in overall accuracy at n = 16 is below 2 points, and smaller than the gap
  at n = 3.

**Against us:** random accuracy at 16 shots more than 1.5 points above n = 6 (more examples keep
helping) or below it (long prompts hurt); or a diverse − random gap at n = 16 of 2+ points. Same
caveats as above: 3 seeds per cell.

## Addendum 3 (2026-11-19, before any Experiment 2 run)

Making Experiment 2 precise before running it; no prediction is changed.

- **Subsets are nested:** one seeded shuffle of the 7,000 training examples (seed 0); N = 1,000 and
  3,000 are the first N of it, so each larger set contains the smaller. The 7,000 point is the
  existing model (best checkpoint by dev loss), re-evaluated through the same vLLM client as the new
  models so all three points share one evaluation path. The 0-shot point is the vLLM 0-shot run of
  Experiment 1 (55.2%).
- **H2.1** is evaluated on overall accuracy with the thresholds as written.
- **H2.2 is operationalized** as: the gain from additional data beyond 1,000 examples,
  acc(7,000) − acc(1,000), is larger for the hard tier than for the easy tier **and** larger for the
  extra tier than for the easy tier. Supported only if both hold by more than 1.5 points (about the
  noise of a single run on a tier of this size, which is larger still for the 168-question extra
  tier; the README will say so when a difference is within noise).
- One training run per size, no seeds: the experiment cannot separate a data-size effect from run-to-run
  training variance of about a point.

## Addendum 4 (2026-11-19, after the training smoke test, before the Experiment 2 runs)

A smoke test of QLoRA on the 8 GB laptop GPU peaked at 5.9 GB with the main run's micro-batch of 4
(×4 accumulation) on typical examples; the 152k-token vocabulary makes the loss tensor large enough
that the longest batches would likely run out of memory. The new models are therefore trained with
**micro-batch 2 × 8 accumulation steps**: the same effective batch of 16 and the same learning rate,
schedule and epochs as the 7,000-example model, which used 4 × 4. This is a small protocol
difference (gradient accumulation changes only the numerical grouping of the same 16 examples) that
is reported with the results, not a change to any hypothesis. Training runs use PyTorch's
`expandable_segments` allocator to limit fragmentation.

## Caveats, stated up front

- One training run per size (no seeds), so size differences under about 1–1.5 points are noise.
- The dev set is used for checkpoint selection (by loss) and for reporting; same caveat as the
  main results.
- A laptop GPU with a small KV cache: throughput numbers are not part of these experiments, only
  accuracy.
