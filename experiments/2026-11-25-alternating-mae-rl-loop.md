# Alternating MAE ↔ RL loop — proposal

**Date:** 2026-11-25 · **Status:** design discussion (not yet implemented)
Follows `2026-11-25-mae-vs-random-5090.md` (the warm-start A/B this builds on).

## Motivation in one paragraph

The A/B showed the trajectory-MAE warm start is worth ~20 of 25 RL steps (~8σ over random init at every eval rung). The natural extension: instead of doing this once, **do it repeatedly** — after each block of RL, self-play with the improved agent, fine-tune the encoder on those better games, then run the next RL block with the trunk frozen so PPO only polishes the two heads. Each cycle upgrades the data the encoder sees and shrinks what RL has to learn.

## The loop

```
BEST = mae_v2_128x16.pth            # A/B-proven warm start
for cycle k in 1..K:
  ① dump    self-play with BEST → 50k trajectory chunks (L=16 frames)
  ② MAE     fine-tune the encoder on those chunks, init from BEST (keep heads)
  ③ RL      frozen trunk, 25 steps × 8192 games; only OutConv + value_head get grads
  ④ gate    if the RL result beat its init at an eval round → it becomes BEST
```

- **Fine-tune, not retrain** — the trunk dims never change, so head weights carry over untouched; fresh optimizer/scheduler each cycle damps oscillation.
- **Dump from BEST, not latest** — the data generator should be the strongest agent we have.
- **Gate** = train.py's own save semantics (`shaza_new.pth` exists ⟺ the agent beat its cycle-init at an eval round). If it didn't, the MAE checkpoint (which inherits the previous heads) becomes BEST anyway — the loop always makes progress.
- Config mirrors the A/B: 128×16, 8192 games/step, seed 42, TEST_FREQ 5, TEST_NUM 600, engine depth 3, EVAL_WORKERS 16. 4 cycles ≈ one overnight on the 5090.

## Why it should reduce training requirements (the math)

This is **block-coordinate descent** over (trunk θ, heads φ):

1. **Frozen-trunk RL is much cheaper per step of progress.** Trainable dim drops from ~14M (θ+φ) to ~0.3M (φ). The backward pass skips the trunk; more importantly, policy-gradient variance scales with the number of trainable parameters — in a linear-features regime, convergence error is ε + O(d_φ/n) instead of O((d_θ+d_φ)/n). Fewer samples needed per unit of improvement.
2. **MAE frames are ~10⁴× cheaper than RL frames.** No credit assignment (per-cell reconstruction + legality + outcome labels, all dense), no interaction with the reward signal's noise. 50k chunks ≈ 20–30 min vs hours for the equivalent RL data.
3. **Each cycle shrinks the representation gap.** The encoder is re-fit on the *current* policy's games, so the features it learns track the data distribution RL actually trains on — the representation error ε decays cycle over cycle instead of accumulating.
4. **Expected net effect:** ~2–4× fewer GPU-hours per rating point, capped by the encoder ceiling (bias–variance tradeoff — the trunk can only learn what the game state space allows).

## Machinery needed (4 pieces)

| piece | what | status |
|---|---|---|
| `scripts/generate_agent_chunks.py` | agent-play dumper: GPU workers load BEST, play with policy sampling (temp 1.0 + ε=0.1 toward uniform-legal), record chronological per-ply (x, mask, outcome ±1), slice 2 random L=16 windows per game, shard like `generate_ssl_chunks.py` | planned |
| `scripts/pretrain_mae.py --init-path` | fine-tune encoder from previous checkpoint (keeps heads) | planned |
| `scripts/train.py --freeze-trunk` | freeze everything except `OutConv` + `value_head`; optimizer filters `requires_grad` | planned |
| `scripts/alternating_loop.sh` | box driver: dump → MAE → RL → gate → repeat; idempotent (crash-resume = re-run) | planned |

## How we'd know it works

Against the A/B's baselines (all at 128×16, seed 42, same eval protocol):

- **4 cycles × 25 frozen steps** vs **100 straight RL steps, unfrozen** (same MAE init) — the loop must beat it on the eval ladder per GPU-hour spent.
- Also comparable: **100 straight RL steps, random init** (the A/B's arm B extended) as the no-pretraining floor.
- If the loop wins, we've measured the compounding effect directly; if it plateaus at the encoder ceiling, we've learned the ceiling — either way it's the number that decides whether alternating is the training recipe.

## Open questions (deliberately deferred)

1. **Cycle length.** 25 steps/cycle is inherited from the A/B. Shorter cycles = fresher MAE data, more MAE overhead; longer = the encoder gets stale. The comparison above is what would tell us.
2. **Chunk volume per cycle.** 50k is a guess — the v2 MAE used 100k random-play chunks; agent games are more informative but less diverse. Worth a sweep (25k vs 50k vs 100k) on the first cycle's MAE validation curves.
3. **Epsilon/temperature in the dumper.** ε=0.1 keeps off-policy coverage for the encoder; too high and the data stops looking like the agent's play. Cheap to A/B via the first cycle's recon metrics.
4. **Gate semantics.** `>50% vs cycle-init` is a weak bar (beating yourself from 25 steps ago is easy) — the real bar is the **vs-fixed-reference** ladder (train.py `--ref-path`, printed as "new vs refernce" every round), which tracks cross-cycle progress on one scale.
5. **Does the trunk actually need re-fitting every cycle, or once?** A single MAE warm start + 100 frozen steps is the cheap ablation — if that matches the full loop, the alternating machinery was unnecessary. One extra arm answers it.
6. **Freeze granularity.** Freezing the trunk mid-run changes the LR dynamics the A/B measured (LR decays 8× over 25 steps and resets each cycle). Head-only training may want its own LR schedule — start with the same one, revisit if the value head lags.
