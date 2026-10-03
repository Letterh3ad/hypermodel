# Decisions

## 2026-10-03 - Goal: research toy built to publishable standard
**Context**: Scope sets how much rigour each stage needs.
**Decision**: Build to learn and understand, but log baselines (e.g. plain LoRA fine-tuning) from every stage so results can be written up. A real-model tool use case may follow if it works.
**Alternatives**: Toy only (cheaper, but adding baselines later is painful); tool-first.

## 2026-10-03 - First domain: synthetic arithmetic
**Context**: Pythia-160M and GPT-2 small score near 0% on GSM8K, which leaves no correct-vs-wrong contrast for the observer.
**Decision**: Synthetic arithmetic with a difficulty dial tuned to roughly 30-60% base accuracy, split into sub-tasks (add, multiply, modular) to form the task stream for the forgetting test. Other domains get tested later.
**Alternatives**: Factual recall edits; GSM8K on a larger model.

## 2026-10-03 - Base model: Pythia-160M
**Context**: Needs to fit an 8 GB laptop GPU and scale within one family.
**Decision**: Pythia-160M, scaling to 410M/1B in the same family.
**Alternatives**: GPT-2 small (more SAE tooling, worse number tokenization).

## 2026-10-03 - Base model replaced: Qwen3-0.6B-Base (supersedes Pythia-160M)
**Context**: Ticket 01 sweep: Pythia-160M and 1B score 0-3% on 2-digit add/mul and copy operands or repeat one answer; their correct/wrong split is chance. Qwen3-0.6B-Base: add-2 100%, mul-2 72%, mod-2 68% (single-digit tokens).
**Decision**: Qwen3-0.6B-Base, scaling 0.6B -> 1.7B -> 4B, the same family as the Heretic models. Pythia SAEs no longer apply; ticket 06 finds Qwen3 SAEs or trains small ones. A small abliterated Qwen3 may be made for the C track when needed. The 4/6/8 layer choice (12-layer Pythia) becomes roughly 1/3, 1/2, 2/3 of Qwen3's 28 layers, re-picked from the ticket 03 sweep.
**Alternatives**: Qwen2.5-0.5B (mul-2 already in band, other family); Pythia (fails the task).

## 2026-10-03 - Mixed operand widths in the difficulty dial
**Context**: On Qwen3-0.6B, mul-2 is 72% and mul-3 is 11%, missing the 30-60% band.
**Decision**: Operands can differ in width (e.g. mul-2x3) to land the main task near 30-50%. mod-2 stays as the second task stream.
**Alternatives**: mul-2 as is.

## 2026-10-03 - Observer inputs: residuals and contrastive first, SAEs second
**Context**: SAE training costs compute and adds bug surface before we know the signal exists.
**Decision**: Step 2 first pass uses raw residuals and contrastive projections. Then add pretrained EleutherAI Pythia SAEs, and train domain SAEs only if the probe needs it. Report each input's contribution as an ablation.
**Alternatives**: Train domain SAEs up front.

## 2026-10-03 - Model-agnostic from day one
**Context**: End goal is making Heretic (abliterated) models self-adaptive, possibly an opsec agent.
**Decision**: Tracing and editing resolve layers through a per-architecture adapter, never hard-coded Pythia module names.
**Alternatives**: Pythia-only code, generalised later.

## 2026-10-03 - Correctness: few-shot, greedy, exact match
**Context**: Small models need examples to answer arithmetic at all; traces must be comparable.
**Decision**: Fixed few-shot exemplar prefix, greedy decode, first integer parsed, exact match.
**Alternatives**: Zero-shot.

## 2026-10-03 - Trace last token and operand tokens at layers 4, 6, 8
**Context**: Storage is cheap at d=768; layer choice needs justifying.
**Decision**: Save those positions at layers 4/6/8, plus a linear-probe sweep across all 12 layers.
**Alternatives**: All tokens, all layers.

## 2026-10-03 - Baselines at every stage
**Context**: Results must stand up to scrutiny.
**Decision**: Step 2: probe on raw residuals, and a bag-of-tokens / operand-size probe. Step 3: plain LoRA at matched rank and layers, plus random-direction edits. Steps 4-6: hand-tuned settings.
**Alternatives**: Add baselines later.

## 2026-10-03 - Tooling
**Decision**: uv + pyproject, raw `transformers` forward hooks, JSONL logs with a plotting script, pytest.
**Alternatives**: TransformerLens/nnsight (Pythia/GPT-centric, more deps), W&B.

## 2026-10-03 - Edit target: residual writers (MLP output projection)
**Context**: Model-agnostic editing needs a role, not a module name. Abliteration also acts on output projections.
**Decision**: Adapter exposes a "residual writer" role; the editor writes the MLP output projection (`dense_4h_to_h`, `down_proj`). Attention output can be enabled later.
**Alternatives**: Attention output only; both from the start.

## 2026-10-03 - Evaluator J as pluggable anchors with floors
**Context**: On a Heretic model, free edits could restore refusals if that scores well.
**Decision**: J is a list of (metric, floor, weight). Arithmetic accuracy only for now; refusal rate and domain evals slot in later.
**Alternatives**: Single Acc_anchor as in the spec.

## 2026-10-03 - Step 1-2 pass bars
**Decision**: Probe on z reaches AUROC >= 0.75 on held-out questions and beats the bag-of-tokens baseline by >= 0.05, mean of 3 seeds. ~5k questions per difficulty, 60/20/20 split.
**Alternatives**: "Clearly better than chance" (unfalsifiable).

## 2026-10-03 - Workflow for steps 1-2
**Decision**: Spec plus tracer-bullet tickets for steps 1-2 only, built on `phase-1-observer`, review gate, then check in on results. Steps 3+ ticketed after.

## 2026-10-03 - Gate for non-arithmetic domains
**Context**: J is the only control on an autonomous self-editor; opsec has no automatic grader.
**Decision**: No step 4+ on any non-auto-gradable domain until it has a sealed, automatically checkable eval. Not designed yet.

## 2026-10-03 - Step 2 bar reframed; "internals beat the question" moves to step 3
**Context**: Tickets 03-04 on mul-2x3: question-only baseline AUROC 0.896, raw residuals 0.900, baseline+raw 0.900, 4 contrastive dims 0.886. With greedy decoding correctness is nearly a function of the question, so "beat the baseline by 0.05" cannot pass however good the observer is.
**Decision**: Step 2 passes if z predicts correctness (AUROC >= 0.75) and is within 0.01 of the best question-only baseline, i.e. the observer recovers unaided what was hand-crafted. Step 3 must ablate the editor conditioned on z vs. on baseline features vs. unconditioned; if z does not help editing, the observer has failed.
**Alternatives**: Keep the bar and fail on arithmetic; switch to sampled decoding for noisier outcomes.
