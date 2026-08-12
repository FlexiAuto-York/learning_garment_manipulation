# System tests

Complete tasks in simulation: spawn a crumpled garment, run a policy, verify the final state
meets the task's success threshold.

Requires a built PyFlex and a GPU, so these do not run in CI. See `../README.md`.

Suggested first test: a short rollout of `sim_exp/magpie/heuristic_centre_sleeve_folding`,
which needs neither a checkpoint nor a dataset.
