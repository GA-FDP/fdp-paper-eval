# FDP Agent Correctness Evaluation — Prompt Set & Gold Pipelines

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21711018.svg)](https://doi.org/10.5281/zenodo.21711018)

Evaluation data for the paper:

> **The Fusion Data Platform and TokSearch: AI-Assisted, IMAS-Native Analysis
> of Tokamak Archives at Scale.** B. Sammuli, M. Clark, S. Denk, S. Jackson,
> S. Smith, K. Lin, T. Bechtel Amara, T. Odstrcil, R. Nazikian.

This repository contains the prompt set, expert-written gold pipelines, and
recorded reference outputs used in the paper's pipeline-generation accuracy
ablation (18 natural-language analysis requests, evaluated on DIII-D over a
locked set of 50 plasma shots from 2024). It is released so the ablation can
be rerun against other models and devices.

## Contents

| File | Description |
|------|-------------|
| `prompts.py` | The 18 natural-language prompts, the locked shot list, and the per-prompt comparison rules |
| `run_gold.py` | The canonical, runnable gold pipelines (executed and verified against the live FDP stack on 2026-06-24) |
| `gold_references.json` | Recorded reference outputs from the verified gold run |
| `source_2024_shots.py` | How the locked 50-shot set was sourced from the DIII-D relational database |

`run_gold.py` is the source of truth for the gold code; `prompts.py` is the
prompt registry the agent harness consumes.

## Reproducing the gold run

The gold pipelines run against the live DIII-D archive via the
[Fusion Data Platform](https://ga-fdp.github.io/). Access to the DIII-D
archive requires authorization and an access token. With an FDP environment
installed (`fdp-install`) and a valid token:

```bash
fdp run python run_gold.py
```

Results are scored against `gold_references.json` using the per-prompt rules
in `prompts.py` (per-shot values within 1% for at least 95% of shots; Jaccard
overlap of at least 0.95 on returned shot sets).

## Scoring criteria

A generated pipeline is counted **correct** only if it executes and its result
matches the reference under the tolerances above; otherwise the dominant
failure mode is recorded. See the paper for the full
protocol, including the C0 (no context) / C1 (documentation over MCP) ablation
conditions.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
