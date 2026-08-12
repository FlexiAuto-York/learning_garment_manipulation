## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## Type

- [ ] Bug fix
- [ ] New capability (controller, arena, task, primitive)
- [ ] Configuration or experiment
- [ ] Documentation
- [ ] Refactor

## Checklist

- [ ] `python -m pytest tests/unit -q` passes
- [ ] Any new config lives under `conf/<group>/<method>/` and its references include the method
      subfolder
- [ ] `exp_name` matches the config filename
- [ ] No machine-specific paths, IPs or device IDs in `.py` files
- [ ] `README.md` and `CHANGELOG.md` updated if method, hyperparameters, metrics or evaluation
      changed

## Verification

<!-- What did you actually run? Paste the command and the outcome. If you ran it in simulation,
     say which config and on what hardware. If you could not verify something, say so. -->

## Effect on existing results

- [ ] Existing checkpoints still load
- [ ] Training behaviour unchanged
- [ ] If either is false, explain below
