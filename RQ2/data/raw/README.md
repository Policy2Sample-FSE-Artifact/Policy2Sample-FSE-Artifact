# RQ2 raw-data manifest

All files here are original JSON outputs copied byte-for-byte from the local experiment output folder; no raw rows were rewritten or aggregated in place. The paper reports the dedicated three-generation AgentSpec runs and the Prompt-only/Policy2Sample rows from the matching model sweeps. The initial five-generation AgentSpec arms remain raw provenance only and are not used for the paper's cross-model comparison.

| File pattern | Contents | Use in current summary |
|---|---|---|
| `results_moving_narrow_window_{2b,4b,8b}.json` | Initial 3-method runs, 33 contexts × 3 methods per model; the embedded AgentSpec arm used a 5-generation budget. | Prompt-only and Policy2Sample rows only. The embedded AgentSpec rows are retained for provenance but are not the AgentSpec values in the current summary. |
| `results_agentspec_budget3_{2b,4b,8b}.json` | AgentSpec `llm_self_examine` only, 33 contexts per model, up to 3 total generations. | AgentSpec rows in the main cross-model comparison. |
| `results_agentspec_budget20_2b.json` | 2B AgentSpec sensitivity run, 33 contexts, up to 20 total generations. | Separate exploratory retry-budget sensitivity result. |

Each JSON contains the Context and oracle safe-speed projection, per-case proposals/trace, final action, safety/no-action outcome, model-call count, and measured latency. See [`../README.md`](../README.md) for run conditions and interpretation limits.
