# Containerised environment

Pins Ubuntu 22.04, CUDA 12.1, Python 3.10 and the `actoris_harena` framework. The repository
itself and SoftGym are bind-mounted rather than copied in, so you edit on the host and run in the
container.

## Build and run

```bash
# allow the container to draw on your X server (once per login)
xhost +local:root

cd docker
docker compose build
docker compose run --rm robot
```

Then inside the container:

```bash
source ./setup.sh
python tool/hydra_eval.py --config-name sim_exp/magpie/heuristic_centre_sleeve_folding
```

## What is not in the image

**PyFlex is not built here.** SoftGym's bindings compile against your CUDA and driver version, so
build them on the host (or in a one-off container run) following SoftGym's own README, then let
the bind-mount carry them in. `docker-compose.yml` expects SoftGym at `../../softgym` relative to
this directory.

**Robot drivers.** The xArm and UR SDKs are installed by `pip install -e .` in the repository, but
network access to the control boxes depends on the host — hence `network_mode: host`.

## Requirements

- NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- `docker compose` v2

## Notes

- `GARMENT_DATA_ROOT` inside the container is `/workspace/data`; on the host it maps to whatever
  `$GARMENT_DATA_ROOT` points at, defaulting to `../data`.
- `privileged: true` and the `/dev/bus/usb` mount exist for RealSense access. Drop both if you only
  need simulation.
