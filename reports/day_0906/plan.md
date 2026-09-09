# Plan — 2026-09-06

Continues the Tier-3 (SOH + monotonicity + Wiener) build from `day_0901`'s
V2 architecture-ladder winner. Today's work: complete Tier-3 5-seed runs for
HUST and CRUSH (MATR1's Tier-3 5-seed was already done pre-session), extend
MATR1 to 7-8 seeds as a robustness check, restructure `paper/paper_0904.md`
around Tier-3 as the primary physics-informed claim (demoting Tier-1 to an
Inter-Cell-Embedding-only role), and fix a CRUSH data-integrity issue
discovered mid-session (see Task #4).

## Scoreboard going in

| Dataset | Status | Ours | README target |
|---|---|---|---|
| MATR1 | ✅ Tier-3 pipeline beats README; Tier-1 numerically best but not adopted as headline (§ paper) | **71.06 ± 5.25** (V1+V2 Tier-3 + Inter-Tier-1, 5-seed, NNLS) | 90 (PCR) |
| HUST | ✅ beaten, but by Tier-1, not Tier-3 | **289.82 ± 12.38** (Tier-1 ensemble, 8-seed) | 322 |
| CRUSH | ❌ open — no configuration beats README or local reproduction | 367.49 ± 9.07 (Tier-3, NNLS, 8-seed) — best ensemble; **354.44 ± 12.21 (V1 solo, Tier-3) is the best single number found, not yet adopted as headline** | 330 |

## Task table (easiest → hardest)

| # | Task | What / why | Est. | Status |
|---|---|---|---|---|
| 1 | HUST Tier-3, 5-seed | **Done.** V1+V2+Inter(Tier-3 embed) NNLS = 313.12±18.55. **Underperforms Tier-1 (289.82±12.38) at every seed and every individual component** (V1, V2, Inter each worse under Tier-3 at all 5 seeds) — not an ensemble-selection artifact. Cross-tier (Tier-3 backbone + Inter-Tier-1) and 4-way variants tested too: best achievable is 308.06±18.02, still short of Tier-1. HUST's proposed result is therefore Tier-1, not Tier-3. | 5 run × ~15-25min | ✅ done |
| 2 | CRUSH Tier-3, 5→8-seed | **Done.** NNLS ensemble: 371.41±6.07 (5-seed) → 367.49±9.07 (8-seed). Simple-mean reverses the ranking at 5 vs 8 seeds (367.59±4.64 vs 370.12±6.33) — flagged as an ensemble-method-instability finding, not resolved in either direction. | 8 run × ~10-20min | ✅ done |
| 3 | MATR1 Tier-3, extend 5→8 seed (robustness check) | **7/8 done**, seed 8 (index 7) in progress. Ordering confirmed stable: Tier-1 ensemble 70.17±5.58 vs Tier-3 74.83±4.55 at n=7 (was 68.93±6.11 vs 73.62±4.85 at n=5) — same conclusion, not a 5-seed artifact. | 3 run × ~15-20min | 🟡 in progress (seed7 mid-training) |
| 4 | CRUSH headline number — data-integrity fix | **Done.** The paper's prior "Baseline 357.18 / Proposed 352.92" pair could not be traced to any `.pkl` in the current, consistent cache (`..._hust10_calce15_fullval_withsoh`, pen=1.4) after an exhaustive search (every `ens_test` value in every seed/config file in the directory). Closest matches were all seed6 across unrelated configs (351.59-354.28) — likely a mislabeled single-seed number in an earlier draft, not a real multi-seed mean. Replaced with verified figures: no-SOH baseline 371.64±5.53, Tier-1 371.29±8.92, Tier-3 367.49±9.07 (all traceable to specific `.pkl` files). **Open sub-finding**: V1 solo under Tier-3 (354.44±12.21) beats every ensemble tried for CRUSH — not yet adopted as the paper's CRUSH headline, pending user decision. | ~1h investigation | ✅ done (correction applied); 🟡 open decision on V1-solo headline |
| 5 | Paper restructure — `paper/paper_0904.md` | **Done**, iteratively across the session. Title changed (dropped "PINN4SOH" as the headline label), Tier-3 promoted to the primary physics-informed claim (real SDE, distinguished from Tier-1's plain auxiliary supervision per Raissi et al. 2019's definition), Tier-1 demoted to "Inter-Cell-Embedding-only" role with the reason stated plainly rather than justified at length, §5.1 table rebuilt as README/local-repro/no-SOH/Tier-1/Tier-3/proposed per dataset, §5.2b (new) added for HUST's full per-seed/per-component breakdown, §5.3 corrected (was mislabeling CRUH data as "HUST-CRUSH"), §6.3 rewritten around the dataset-dependent (and in HUST's case, negative) Tier-3 result, §7 Limitations rewritten with 7 disclosed items including the CRUSH data-integrity correction and the λ_SOH test-informed-screening disclosure (λ=0.01 for MATR1, λ=0.10 for HUST/CRUSH). | ~2h, iterative | ✅ done |

## Standing directions

- **Tier-3 (SOH + monotonicity + Wiener) is the paper's primary physics-informed claim**, not Tier-1 (SOH-only auxiliary supervision) — per user's explicit instruction ("chúng ta dùng Tier3 làm point chính trong model, ensemble thì mới đề cập thêm Inter-Embedding Tier1"). Tier-1's only role in the final pipeline is as the embedding source for Inter-Cell Embedding (empirically better there than a Tier-3 embedding — see `day_0901`-adjacent session notes and `paper_0904.md` §6.2).
- **The best-performing tier is dataset-dependent and reported as such, per dataset** — never force one method's number into every dataset's "proposed" cell. MATR1 → Tier-3-based pipeline (within noise of Tier-1's own best). HUST → Tier-1 (Tier-3 measurably hurts, all 5 seeds, all components). CRUSH → Tier-3 ensemble is the current headline, but V1-solo-Tier-3 (354.44±12.21) is a stronger, still-undecided candidate.
- **Any historical result number must be traceable to a specific `.pkl`/log before being cited** — the CRUSH 352.92/357.18 incident (Task #4) is the concrete cautionary case; when a number can't be found by direct search, say so and replace it, rather than assuming it was a typo of something close.
- **Ensemble-method choice (NNLS vs. simple-mean) is not settled for small-val-set datasets** (CRUSH val n=15) — the two methods' relative ranking reverses between 5 and 8 seeds; do not pick whichever flatters a single result.

## Open items carried from day_0901

1. Tier-0.5/1/2/3 PINN screen on the `day_0901` architecture-ladder V2 backbone (Task #7 there) — superseded by this session's full Tier-3 build on the axis-aware V1/V2/Inter-Embedding architecture; no longer tracked separately.

## Open items from today

1. **MATR1 seed7 (8th seed) still running** — finish and fold into the robustness-check numbers in `paper_0904.md` if it changes the picture (unlikely given seeds 0-6's consistency).
2. **CRUSH "Proposed" headline decision**: adopt V1-solo-Tier-3 (354.44±12.21, beats every ensemble tried) as the paper's CRUSH result, or keep the NNLS ensemble (367.49±9.07) for methodological consistency with MATR1/HUST (which both report ensembles, not solo components) — needs an explicit user call, not a default.
3. **SNL, CALCE, CRUH single-seed screens** — still not run; §5.4 of the paper is a placeholder pending these, with the too-small-val-set limitation to be stated explicitly per prior user instruction.
4. **HUST's negative Tier-3 result mechanism is still a hypothesis, not confirmed** — CHUNK=48 chunking was floated as an explanation but CRUSH uses the same chunk size without showing the same degradation, so the mechanism remains open (`paper_0904.md` §6.3).

## Standing constraints

- **Sequential GPU execution only** — never two training jobs at once (4GB GPU). Reconfirmed necessary again today: several background jobs were silently killed by Claude Code session restarts (not code bugs, not OOM) mid-session; the fix is always to relaunch, since training is fully deterministic and reproduces identically up to the kill point.
- **Multi-seed discipline**: never claim a result off fewer seeds than declared for that dataset (5 for MATR1/HUST/CRUSH main results); a single-seed number is a screen, stated as such.
- **Pre-declared hyperparameters**: λ_SOH=0.01, λ_mono=0.05, λ_wiener=0.05, τ=0.002 fixed across all three datasets for the Tier-3 backbone. The one disclosed deviation (λ_SOH screened per-dataset for the Tier-1 embedding source, §4.3/§7 of the paper) is reported, not hidden.
- **Benchmark bar**: README numbers are the target (MATR1=90, HUST=322, CRUSH=330), not locally-reproduced baselines — except where a genuine reproducibility gap exists (CRUSH: local repro=355, a real, disclosed gap from README).
