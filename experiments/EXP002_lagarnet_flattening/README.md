# EXP002 — LaGarNet garment flattening via latent-dynamics MPC

## Purpose

Train a goal-conditioned latent dynamics model (RSSM) offline on flattening trajectories, then use
it for model-predictive control to flatten crumpled garments. Second golden path; contrast with
EXP001, which learns a policy directly rather than a model.

## Setup

| | |
|---|---|
| Robot | Single simulated picker, PyFlex/SoftGym |
| Garments | Long-sleeve (transfer configs cover trousers, skirt, dress) |
| Observation | 64×64 RGB + goal, with mask |
| Action | `norm-pixel-single-pick-and-place` (4 params) |
| Config | `sim_exp/lagarnet/final_lagarnet_40000_eps` |
| Dataset | `all_garment_mix` (40k episodes) |
| Release | v0.1.0 |

## Method

Recurrent state-space model with a 300-d deterministic and 60-d stochastic latent, trained offline
(`train_mode: offline`) with a coverage-alignment reward processor (5 reward layers, overshooting
scale 0.1). At evaluation the learned model plans pick-and-place actions by rolling out candidates
in latent space (`SingleArmMaskPickAndPlaceMPC`).

## Reproduce

```bash
source ./setup.sh
export GARMENT_DATA_ROOT=/path/with/space
./run.sh
```

## Metrics

Normalised coverage, max IoU against the flattened goal, success rate.

## Variants shipped

The ablations behind the paper's tables are all present as sibling configs:

- Dataset size: `final_lagarnet_{100,500,1000,10000,40000}_eps`
- Reward design: `final_lagarnet_{cf,l2u,pc,sfa}_reward`, `final_lagarnet_reward_v*`
- Planning horizon: `tool/hydra_horizon_ablation.py`

## Known limitations

- Requires the `all_garment_mix` dataset, not yet publicly hosted (README §4).
- Flattening only; folding is covered by MAGPIE (EXP001).
- MPC evaluation is considerably slower than a feed-forward policy — expect long evaluation runs.
