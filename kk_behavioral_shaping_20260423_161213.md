# KK Behavioral Shaping - 20260423

Remote output root: `/u701/tevillen/rl-takehome/output/kk_behavioral_shaping_20260423_161213`

## Motivation

Earlier KK-trained runs did not improve KK high-k performance and sometimes compressed breadth badly. This sweep tested whether CRL could still be useful as a behavior-shaping tool: not just "does it improve the final score?", but "can changing CRL pair construction move the model toward different output behavior?"

## Hypothesis

The working hypothesis was that KK needs more behavior-aware CRL pairs than the baseline correct-vs-incorrect setup. We expected different pair constructions to move the model differently: gold anchors might sharpen top-1 answer format, implicit length/budget shaping might shorten outputs, hard negatives might improve discrimination, and a decayed old+new buffer might preserve more breadth than an unweighted rollout mix.

## Experiment

All runs used the same footprint:

`GRPO-100 -> collect_rollouts(450) -> CRL-5 -> GRPO-100`

Common settings:

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Train dataset: `knights_knaves`
- KK config: `5 people, depth=2, width=2`
- Seed: `7111997`
- CRL LR: `2e-6`
- CRL batch size: `32`
- Eval: `100` problems, `64` samples, `pass@1/16/64`
- Eval datasets: KK and Countdown

| Run | CRL change | Merge old+new? | Old rollout weight |
|---|---|---:|---:|
| `implicit_budget` | Short correct rollouts as positives; long correct rollouts as negatives | no | `1.0` |
| `anchor_inject` | Gold answer strings used as CRL positives | no | `1.0` |
| `hard_neg_miner` | Negatives restricted to structured near-misses | no | `1.0` |
| `decay_buffer` | Baseline CRL with old+new merged rollout data | yes | `0.35` |

## Results

| Run | KK p@1 | KK p@16 | KK p@64 | Countdown p@1 | Countdown p@16 | Countdown p@64 | Final GRPO avg seq len, last 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `implicit_budget` | `0.0598` | `0.2731` | `0.4100` | `0.0192` | `0.1603` | `0.3200` | `27.77` |
| `anchor_inject` | `0.0669` | `0.2489` | `0.3400` | `0.0173` | `0.1401` | `0.2700` | `28.56` |
| `hard_neg_miner` | `0.0647` | `0.2629` | `0.3800` | `0.0167` | `0.1431` | `0.2700` | `27.92` |
| `decay_buffer` | `0.0636` | `0.2841` | `0.4200` | `0.0166` | `0.1504` | `0.2900` | `28.05` |

Baseline context:

| Run | KK p@1 | KK p@16 | KK p@64 | Countdown p@1 | Countdown p@16 | Countdown p@64 |
|---|---:|---:|---:|---:|---:|---:|
| `A_kk_grpo225` | `0.0384` | `0.2759` | `0.4825` | `0.0220` | `0.1811` | `0.3291` |
| `D_kk_new_crl` | `0.0430` | `0.2049` | `0.3139` | `0.0212` | `0.1752` | `0.3151` |

## Conclusion

These runs show that CRL pair construction is steering behavior, even when the final high-k score is not yet better than pure GRPO. `anchor_inject` produced the best KK p@1, consistent with the idea that gold-answer anchors sharpen top-1 format/behavior, but it also lost the most p@64 breadth. `decay_buffer` produced the best KK p@16/p@64 among the shaping variants, and its merged CRL file had `265` usable rollout records, making it the strongest behavioral variant for preserving breadth. `implicit_budget` had the best Countdown transfer and slightly shorter final GRPO sequence lengths, but the length effect was small rather than a dramatic compression.

The right interpretation is not "these variants failed." It is that CRL is movable: different pair definitions produce different tradeoffs. KK continues to behave unlike Countdown: CRL variants can improve top-1 behavior while compressing high-k breadth, which supports the hypothesis that KK has a smaller, more template-like answer manifold where CRL can concentrate probability mass too aggressively.

## Current Read

The most useful comparison is still against the earlier `A_kk_grpo225` and `D_kk_new_crl` baselines. `D_kk_new_crl` improved p@1 but badly hurt p@64, while these shaping variants partially recover high-k breadth without beating pure GRPO:

| Run | KK p@1 | KK p@64 | Read |
|---|---:|---:|---|
| `A_kk_grpo225` | `0.0384` | `0.4825` | Best high-k breadth among available KK-trained runs |
| `D_kk_new_crl` | `0.0430` | `0.3139` | Top-1 improves, breadth collapses |
| `anchor_inject` | `0.0669` | `0.3400` | Strongest top-1 sharpening, still narrow |
| `implicit_budget` | `0.0598` | `0.4100` | Better breadth than baseline D, best Countdown transfer |
| `hard_neg_miner` | `0.0647` | `0.3800` | Good top-1, middle of the pack on breadth |
| `decay_buffer` | `0.0636` | `0.4200` | Best shaping variant for p@16/p@64 |

## Qualitative Follow-Up

The next qualitative check should measure whether these CRL modes changed answer diversity rather than just score. For each model, sample KK at `n=128` and compute:

- unique normalized assignments per prompt
- modal assignment frequency
- entropy over normalized assignments
- average Hamming distance between predicted and gold role assignments
- completion length distribution

The prediction from the current results is that `anchor_inject` will have the highest modal mass and lowest diversity, while `decay_buffer` and `implicit_budget` should retain more unique assignments. This would directly test the "top-1 sharpening vs high-k breadth" explanation.

For future runs, `anchor_inject` is worth testing with a longer GRPO recovery phase to see whether p@k breadth recovers after the top-1 sharpening. `hard_neg_miner` and `decay_buffer` are also plausible longer-run candidates. We do not yet have a clean apples-to-apples no-decay `E` run for KK, so `decay_buffer > E` cannot be claimed from this batch.
