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

## 2026-10-03 - Step 2 verdict is a non-inferiority test on 20k traces
**Context**: Gate round 1: the 3-seed spread ignored test-sampling error (1000 test rows); paired bootstrap of z minus baseline had CIs down to -0.025, and an observer-sized MLP on question-only features matched z.
**Decision**: z is scored by its own head on test rows. Pass needs AUROC >= 0.75 and, for every seed, the lower 95% paired-bootstrap bound of (z - linear baseline) >= -0.01. Report the MLP-on-baseline control. Traces re-recorded at 20k (4000 test rows) to halve the CI width. Splits keyed so a*b and b*a share a side; contrastive bases and SAE latent choice fit on train only.
**Alternatives**: Mean over seeds against a point threshold (what round 1 broke).

## 2026-10-03 - Step 3 editor: shared LoRA bank with z-conditioned mixing
**Context**: A raw hypernetwork emitting U, V (rank 8, ~32k outputs per layer) is large and unstable.
**Decision**: Per edited layer, a learned bank of E rank-r LoRA experts on the MLP output projection; the editor maps the conditioning vector to per-question mixing weights and scales. Unconditioned mode (constant mixing) is plain LoRA at matched parameter count and serves as the LoRA baseline. Edits are transient (per question, discarded).
**Alternatives**: Raw hypernetwork to U, V (MEND-style).

## 2026-10-03 - Step 3 placement, observer training, retain set
**Decision**: Edit layers 12, 15, 18, 21 at rank 8 (later tunable in step 5). Observer initialised from step 2 and fine-tuned end to end at a lower LR; frozen-observer variant as ablation. Retain set: generic text snippets plus add-4 and mod-3x1 questions.
**Alternatives**: Edit all layers; frozen observer only; text-only retain set.

## 2026-10-03 - Step 3 pass bar
**Decision**: On held-out mul-2x3 (4000 test questions, 3 seeds): z-conditioned editor raises accuracy above base (33%); retain KL <= 0.05 nats/token; z-conditioned beats both plain LoRA (unconditioned) and question-feature-conditioned with paired-bootstrap CI excluding zero. Failing the last is a valid research result, not something to tune away.
**Alternatives**: Accuracy gain alone (would not test the observer).

## 2026-10-03 - Step 3 training defaults (ticket 01)
**Context**: Plain-LoRA baseline needed fixed hyperparameters before the conditioned variants reuse the loop.
**Decision**: Target is answer + newline (prompt masked), matching the scorer's stop. AdamW lr 1e-3, batch 16, alpha 16 (scale 2), constant mixing initialised to 1, V Kaiming, U zero. Early stopping scores 1000 sampled val questions every 250 steps, keeps the strictly best checkpoint, patience 4, cap 3000 steps. Every variant uses the same settings.
**Alternatives**: Stop on val loss (cheaper, but accuracy is the measured outcome); tune per variant (would confound the ablation).

## 2026-10-03 - Retain KL bar applies per source; training KL balances sources
**Context**: A paragraph has ~10x the scored tokens of an arithmetic question, so the pooled per-token KL is mostly text. Plain LoRA trained without retain scored pooled 0.018 (passes 0.05) while add-4 was 0.081 and mod-3x1 0.111.
**Decision**: (F's call) The 0.05 nats/token bar applies to each retain source (text, add-4, mod-3x1); `evaluate_retain` reports each and `worst`. The training KL term is the mean of per-source token means, on stratified batches of 3 items per source. Edit-norm gamma defaults to 0.1 (about 4% of answer CE at the plain-LoRA edit size).
**Alternatives**: Pooled tokens (bar tests almost nothing); mean of sources (lets one source exceed 0.05).

## 2026-10-03 - Step 3 conditioning pipeline (ticket 03)
**Context**: The editor needs z for every input it edits, including retain items, and the spec's raw+contrastive observer was never saved.
**Decision**: z variants use the step 2 `raw` observer (z AUROC 0.910, same as `all`, no SAE). The unedited pass is deterministic, so conditioning inputs are computed once per input: task questions reuse the phase 1 traces, retain items are traced at startup. Arithmetic retain items are read at their operands; generic text at its last token in all three slots. Every conditioned mixer is g0 + head(c) with a zero-init last layer, so it starts exactly at plain LoRA. The fine-tuned observer trains at lr / 10 and its correctness head is frozen. Question features use the phase 1 baseline schema over all arithmetic items plus an is-text flag (text rows otherwise 0).
**Alternatives**: Retrain a raw+contrastive observer first; recompute traces every step (same values, slower); skip conditioning for retain items (would hide what the editor does off-task).

## 2026-10-03 - Retain arithmetic pool is seed-independent
**Context**: Retain test items are fixed (seed 0); training for seeds 1 and 2 drew their arithmetic pool with their own seed, which could overlap the test tail.
**Decision**: One pool per setting, generated with seed 0: train takes the head, test the tail, for every seed. Seed varies only the generic-text sample.
**Alternatives**: Per-seed test sets (verdict rows would differ across seeds).

## 2026-10-03 - Hidden-state router baseline (MoLE-style)
**Context**: A reviewer would ask whether the observer's separate unedited read (z) beats the standard mixture-of-LoRA-experts recipe, a router reading the current hidden state in the same pass.
**Decision**: Fifth conditioning kind `router`: per edited layer and per token, g = g0 + W_l h + c_l, h being that writer's own input (the MLP activation fed to down_proj) in the edited pass. W, c zero-init so it starts at plain LoRA; linear and unnormalised like the other mixers, so g can be any sign. `LoRABank.apply` takes either a mixing tensor or a router called in each hook as router(l, h) -> [B, T, E]; a mixer's output is passed to apply as before, so call sites are unchanged. Cached generation routes each new token on its own h. Qwen3-0.6B (d_in 3072, 4 layers, E=8): 98,368 router params (W 98,304, g0 32, c 32).
**Alternatives**: Softmax or top-k gating (changes the edit scale versus the other kinds); routing on the residual stream (another hook site); route once per sequence from the prompt (not the standard baseline).

## 2026-10-03 - Editor question features exclude the answer's magnitude
**Context**: Phase 1's baseline features include log10(answer), harmless for predicting correctness but answer-derived information when fed to an editor (float32 resolves a 5-digit answer).
**Decision**: (F's call) The features variant uses operand-only features (`baseline_features(..., with_answer=False)`); phase 1 keeps the original set. The first features run was killed 5 min in and restarted.
**Alternatives**: Keep it as a stronger-than-fair control.

## 2026-10-03 - Step 3 ablation runs to a fixed step cap
**Context:** Seed-0 features run early-stopped at 2250 (patience 4) mid-learning; plain LoRA hit the 3000 cap still rising. Val (1000 q) noise is ~1 pt, so patience 4 cut variants unevenly.
**Decision:** Ticket 04 runs every variant with `--max-steps 5000 --patience 20` (cap decides); best-val checkpoint restored as before.
**Alternatives:** keep patience 4 (unfair to slow starters); drop early stopping code (unneeded, high patience is equivalent).

## 2026-10-04 - Router gates read unit-norm writer inputs
**Context:** Seed-0 router run never learned (test 0.334 = base): gates were `g0 + h @ W + c` on raw Qwen MLP activations, so small W steps swung per-token gates wildly.
**Decision:** `RouterMixer.route` L2-normalises h before `@ W` (parameter-free), bounding each W step's effect on a gate by about lr * sqrt(d_in). Same LR as the rest.
**Alternatives:** lower router LR (still scale-dependent per layer); bounded gates via tanh/softmax (changes the plain-LoRA starting point and MoLE comparison); RMSNorm (same direction, sqrt(d_in) larger steps).

## 2026-10-04 - Shuffled-z control and trimmed step 3 ablation
**Context:** Seed 0: z-finetune ties plain LoRA (0.467 vs 0.468, paired CI [-1.1, +1.0] pt) with ~10x lower retain KL, but has 3.4M trainable params vs 1.05M. Features and z-frozen trail. 15 runs at 5000 steps is ~2.5 GPU-days.
**Decision:** Add `--conditioning z-shuffled`: z-finetune with trace rows permuted (fixed seed) within task rows and within retain rows, so capacity and input distribution match but per-input information is gone. Ticket 04 trimmed to: router recheck (seed 0), z-shuffled + z-finetune seed 0 at 5000, then plain + z-finetune seeds 1-2.
**Alternatives:** shuffle across task and retain rows (also removes arithmetic-vs-text identity, which features already give, so it would conflate two effects); full 15-run grid (mostly confirms no accuracy gain).
