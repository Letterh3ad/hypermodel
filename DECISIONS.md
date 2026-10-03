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
