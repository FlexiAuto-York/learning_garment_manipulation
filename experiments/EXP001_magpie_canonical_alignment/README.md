# EXP001 — MAGPIE canonicalisation-alignment from crumpled states

## Purpose

Train and evaluate the MAGPIE multi-primitive flow-matching policy on canonicalisation-alignment:
bringing a crumpled garment into a canonical, aligned configuration. This is the headline result and
the repository's golden path — if this runs, the installation is correct.

## Setup

| | |
|---|---|
| Robot | Two simulated pickers (dual-arm), PyFlex/SoftGym |
| Garments | All simulated types: long-sleeve, trousers, skirt, dress |
| Observation | 128×128 RGB + goal RGB, with mask |
| Action | 4 primitives — pick-and-fling, dual pick-and-place, single pick-and-place, no-op |
| Config | `sim_exp/magpie/magpie_ctr_align_all_sim_garments_p4_v126_hindsight` |
| Dataset | `all_garments_multi_primitive_alignment` (human multi-primitive demos) |
| Release | v0.1.0 |

## Method

ResNet-18 + GroupNorm encoder → 512-d embedding. An MLP classifier (`[256,128]`, LayerNorm, dropout
0.5) selects one of K=4 primitives. Per-primitive `ConditionalMLP1D` flow-matching heads
(`[1024,1024,1024,512]`, GELU, dropout 0.1) generate the continuous parameters, conditioned by
concatenation, with 4-step Euler inference at test time. Loss is optimal-transport conditional flow
matching. Goals are relabelled in hindsight (`HindsightDataset`, 0.8 future-goal probability).

AdamW, lr 3e-4, weight decay 1e-2, batch 1024, 120k updates, 3k warmup, EMA 0.9999.

## Reproduce

```bash
source ./setup.sh
export GARMENT_DATA_ROOT=/path/with/space
./run.sh                 # train, then evaluate
```

Expect roughly 24 GB of GPU memory and a day of training on a single modern GPU.

## Metrics

Normalised coverage, max IoU against the goal state, and success rate under the task's IoU
thresholds. Reported by `tool/hydra_eval.py` and logged to W&B.

## Known limitations

- Requires the `all_garments_multi_primitive_alignment` dataset, which is not yet publicly hosted
  (README §4). Without it, only evaluation from a released checkpoint is possible.
- Alignment only. Folding on top of alignment is the
  `canonicalisation_alignment_centre_sleeve_folding` task, which chains this policy with a folder
  via `iou-based-stitching-policy`.
- Trained and evaluated in simulation. Real-world transfer needs the domain-adapted configs in
  `conf/real_world_exp/`.
