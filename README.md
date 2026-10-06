# Hypermodel

A research project on whether a model's own internal activity can tell it how to fix itself.

A small second network (the **observer**) reads a base model's hidden states while it answers a question.
An **editor** turns what the observer saw into a temporary low-rank edit of the base model's weights, and the
edited model answers again. Later stages make edits persist and let the hypermodel edit itself on a slower
clock, with every self-edit gated by a fixed evaluator it cannot change.

The current testbed is deliberately small: **Qwen3-0.6B-Base** on synthetic multiplication (2-digit x 3-digit,
few-shot, greedy, exact match). The base model gets **33.5%** of held-out questions right.

## The question each step tests

| Step | Question | Status |
|---|---|---|
| 1. Traces | Record residual-stream states at the operand and final tokens (layers 9, 14, 19). | done |
| 2. Observer | Can a learned code `z` (64 dims) read from those states predict whether the answer will be right? | done |
| 3. Transient editor | Does conditioning a per-question weight edit on `z` beat edits that do not look at the model's internals? | done: **no** (tie) |
| 4. Persistent editor | Accumulate edits over a task stream, accepted only if a fixed evaluator approves. | next |
| 5-6. Self-tuning, self-editing | The hypermodel sets its own hyperparameters, then edits its own weights, through the same gate. | planned |
| 7. Scale up | Repeat on a larger model. | planned |

### Step 2 result

`z` predicts correctness with AUROC **0.910** on 4000 held-out questions, matching hand-made question features
(0.900 linear, 0.906 MLP). Under greedy decoding, correctness is almost a function of the question, so the
observer recovering that unaided is the pass condition. Whether `z` carries anything *beyond* the question
is the job of step 3.

## Step 3: what is being compared

Every variant trains the same small adapter: a bank of 8 rank-8 LoRA experts on the MLP output projection of
layers 12, 15, 18 and 21 (about 1M parameters), with the base weights frozen. Training minimises answer loss
plus a **retain** penalty, the KL divergence from the unedited model on unrelated text and other arithmetic,
so the edit cannot quietly break everything else. The variants differ only in what picks the mix of experts:

| Variant | Mixing chosen by | What it controls for |
|---|---|---|
| `none` | a fixed learned mix (plain LoRA fine-tuning) | the standard baseline |
| `features` | the question's operands | "it only needs to know the question" |
| `z-frozen` | `z` from the step 2 observer, observer frozen | does the pretrained read help as is? |
| `z-finetune` | `z`, observer fine-tuned end to end | the main hypothesis |
| `z-shuffled` | `z-finetune` with each input given another input's trace | is it `z`'s information, or just extra trainable capacity? |
| `z-gain` | `z` sets one strength per layer instead of the expert mix | does `z` help if it only says *how much* to edit? |
| `z-gain-shuffled` | `z-gain` with shuffled traces | the same capacity control for `z-gain` |
| `router` | the current hidden state at each token (mixture-of-LoRA-experts style) | the standard per-token routing recipe |

**Pass bar** (fixed before the results): on 4000 held-out questions over 3 seeds, the `z` editor beats base
accuracy, keeps retain KL at or below 0.05 nats/token on every retain source, and beats both `none` and
`features` with a paired-bootstrap 95% CI that excludes zero. Failing the last part is a valid result.

### Results

4000 held-out questions. Differences are paired by question (the same questions, answered by both runs), with
95% bootstrap CIs. `none` and `z-finetune` ran 3 seeds; the rest are seed 0.

| Variant | Test accuracy | vs plain LoRA (pts, 95% CI) | Worst retain KL | Trainable params |
|---|---|---|---|---|
| base model | 0.335 | | | |
| `none` (plain LoRA) | 0.468 / 0.475 / 0.465 | | 0.0020 | 1.05M |
| `z-finetune` | 0.457 / 0.471 / 0.465 | -0.47 [-1.18, +0.18] pooled | 0.0003 | 3.43M |
| `z-frozen` | 0.461 | -0.65 [-1.82, +0.53] | 0.0015 | 1.05M |
| `z-gain` | 0.463 | -0.50 [-1.62, +0.62] | 0.0002 | 3.43M |
| `z-shuffled` | 0.458 | -1.00 [-2.17, +0.10] | 0.0002 | 3.43M |
| `z-gain-shuffled` | 0.474 | +0.62 [-0.47, +1.75] | 0.00004 | 3.43M |
| `router` | 0.446 | -2.17 [-3.33, -1.00] | 0.0099 | 1.15M |
| `features`* | 0.424 | -4.38 [-5.58, -3.20] | 0.0023 | 1.06M |

\* `features` and `router` stopped at 3000 steps; the others ran to 5000.

![Step 3: accuracy difference from plain LoRA with 95% CIs, and worst retain KL, per variant](assets/step3-results.png)

**Verdict: no.** Every editor lifts accuracy by about 13 points over the base model, but no variant that reads
`z` beats plain LoRA, so step 3 fails its pass bar. The controls say why:

- **`z`'s content is unused.** Swapping each question's trace for another question's changes nothing:
  `z-finetune` vs `z-shuffled` is -0.05 [-1.20, +1.10]. In gain mode the real `z` is slightly *worse* than a
  shuffled one (-1.12 [-2.10, -0.12], one seed, one of several comparisons, so weak evidence).
- **The conditioning is not collapsed.** The expert mix really does vary from question to question (spread
  0.6 to 1.6, against a pre-registered collapse threshold of 0.1; 6 to 8 effective experts out of 8). The editor
  changes its edit per question, but those changes do not track what the model gets wrong.
- **The lower side effects come from the architecture, not from `z`.** The `z`-conditioned editors drift about
  10x less on unrelated inputs, but so do the shuffled controls.
- A second observer input (logit-lens conflict signals between layers) did not predict correctness better than
  `z` alone, so its editor run was skipped by a pre-registered rule.

This is a negative result for the specific claim "an observer's read of the hidden state makes a better
per-question weight edit than plain fine-tuning", at this model size and on this task. Under greedy decoding,
correctness here is almost a function of the question (step 2), which leaves `z` little to add.

### What next

Step 4 changes the question from "edit better per question" to "accumulate edits over time": a generator
writes one new rank-8 expert per episode from the observer's view of recent failures, over a stream of
arithmetic tasks, and a fixed evaluator decides which experts stay. The full system has to beat naive
sequential LoRA on final stream accuracy (paired CI excluding zero, 3 seeds).

Because step 3 tied, step 4 starts with a cheap pilot fixed in advance: on one task, does a generated expert
beat a randomly started one given the same refinement budget? If yes, build the rest. If not, the hypernetwork
line stops here.

## Running it

```bash
uv sync
uv run pytest                      # full suite, ~2-3 min; -m "not slow" for the fast subset
uv run python -m hypermodel.edit_train --traces runs/traces-mul-2x3-20k \
    --conditioning z-finetune --out runs/edit-z-finetune-s0
uv run python -m hypermodel.watch  # live training curves for every runs/edit-* directory
```

Traces and checkpoints (`runs/`) are not in the repo; `hypermodel.trace` and `hypermodel.observer` regenerate
them. One 8 GB GPU is enough for everything above. Set `CUDA_VISIBLE_DEVICES=-1` to test on CPU.

## Layout

`src/hypermodel/`: `arithmetic` (task and difficulty dial), `adapter` (model-agnostic hooks), `scoring`,
`trace`, `probe` / `contrastive` / `sae` / `observer` (step 2), `editor` (LoRA bank), `condition` (mixers),
`retain` (retain set and KL), `edit_train` (step 3 train/eval CLI), `conflict` (logit-lens conflict signals),
`mix_usage` (how much a conditioned mix varies), `compare` (paired-bootstrap run comparison), `decide`
(pre-registered queue rules), `watch`.

Design decisions and their alternatives are logged in [`DECISIONS.md`](DECISIONS.md).
