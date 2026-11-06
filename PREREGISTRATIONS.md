# Pre-registrations

Every hypothesis test, with its bar as fixed before data, any amendment (dated, and made before the runs it
affects) and the verdict. Measurement-only runs have no bar and are listed for completeness. Rationale and
alternatives for each are in [`DECISIONS.md`](DECISIONS.md).

## Step 2: observer predicts correctness

- **Bar (2026-10-03):** probe on `z` reaches AUROC >= 0.75 on held-out questions and beats a bag-of-tokens baseline
  by >= 0.05, mean of 3 seeds.
- **Amendment 1 (2026-10-03, after tickets 03-04 measured baselines, before the step 2 verdict):** a question-only
  baseline reached 0.896. Under greedy decoding correctness is nearly a function of the question, so "+0.05" cannot
  pass however good the observer is. New bar: AUROC >= 0.75 and within 0.01 of the best question-only baseline.
  The "internals beat the question" test moves to step 3.
- **Amendment 2 (2026-10-03, after review gate round 1):** the 3-seed mean ignored test-sampling error. New bar:
  AUROC >= 0.75 and, for every seed, the lower 95% paired-bootstrap bound of (`z` - linear baseline) >= -0.01.
  Traces re-recorded at 20k (4000 test rows); splits keyed so `a*b` and `b*a` share a side.
- **Verdict: pass.** AUROC 0.910 vs 0.900 linear / 0.906 MLP on question features.

## Step 3: `z`-conditioned editor beats plain LoRA

- **Bar (2026-10-03):** on held-out mul-2x3 (4000 test questions, 3 seeds) the `z`-conditioned editor raises
  accuracy above base (33%), keeps retain KL <= 0.05 nats/token on every retain source, and beats both plain LoRA and
  a question-features editor with paired-bootstrap 95% CI excluding zero. Failing the last part is a valid result.
- **Added control (2026-10-04, after seed 0 tied):** `z-shuffled` (trace rows permuted within task and retain rows)
  to separate `z`'s information from its extra capacity.
- **Go/no-go rules (2026-10-04, fixed before data):** the conflict-signal observer gets an editor run only if its
  AUROC beats the raw observer with CI > 0 on every seed; sparse top-k gating runs only if `z-finetune`'s mix spread
  is < 0.10.
- **Verdict: fail (tie).** `z-finetune` vs plain LoRA -0.47 pts [-1.18, +0.18]; real vs shuffled `z` -0.05
  [-1.20, +1.10]. Both follow-ups skipped by their rules.

## Step 4 pilot: generated expert beats a standard LoRA init

- **Bar (2026-10-13, before any training code):** mul-2x3 stream episode 0, empty bank, 3 seeds. Arms: (A) generated
  expert + 50 refine steps; (B) standard LoRA init + the same 50 steps, same data order; (C) generated, no refine.
  Metric: greedy accuracy on the sealed 500-question mul-2x3 gate. Pass: A - B paired bootstrap pooled over seeds
  (5000 resamples), 95% CI lower bound > 0. Pass builds the rest of step 4; fail stops the hypernetwork line.
- **Amendment (2026-10-13, before any meta-training run):** meta-training settings add-3 and mul-1x3 (base accuracy
  .97, too few failures to learn from) replaced by add-8 (.62) and mod-5x1 (.30). Still disjoint from the stream by
  operand width.
- **Verdict: fail.** A - B +1.33 pts [-0.33, +3.07]. Hypernetwork line stopped. Context ablation and longer refines
  were run afterwards as exploratory only.

## Step 4 full system (not run)

- **Bar (2026-10-04):** full system beats naive sequential LoRA on final average stream accuracy, paired bootstrap
  CI excluding 0, 3 seeds; forgetting reported per task. Not run, because the pilot failed.

## Measurement only (no bar)

- **2026-10-13:** sequential LoRA with and without retain/norm penalties (3 seeds each). The penalised run is the
  step 4 baseline.
- **2026-10-14:** continual-learning reference points: joint, oracle per-task bank, replay (3 seeds each). Written as
  measurement-only before any run.
