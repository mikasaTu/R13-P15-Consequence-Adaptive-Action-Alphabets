# R13-P15 CAAA-v2 — LIBERO Stage 1

This repository implements the oracle consequence-geometry audit for
Consequence-Riemannian Action Alphabets (CAAA-v2) on four standard LIBERO-Goal
tasks. It does **not** train a policy.

The formal artifact root is:

```text
/mnt/cpfs/zbl-cpfs-new/dataset/leon/experiments/r13_p15_caaa_v2/stage1
```

The simulator is standard LIBERO (not LIBERO-Plus perturbations) at the source
tree corresponding to upstream commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
The official `libero_goal` demonstrations are read from the new CPFS dataset
namespace. All branches use Panda `OSC_POSE`, 20 Hz, and `H=4`. The 24
continuous chunk coordinates are the six delta-pose channels at four control
steps; the gripper command is frozen to the demonstration and copied exactly.

Local smoke:

```bash
MUJOCO_GL=glx \
PYTHONPATH=/mnt/cpfs/zbl-cpfs-new/USERS/leon/code/LIBERO-original \
/mnt/cpfs/zbl-cpfs-new/USERS/leon/envs/libero-original/bin/python \
  -m caaa_libero.cli smoke --output-root /tmp/caaa_smoke
```

The PAI launcher performs the same stages with durable, atomically completed
snapshot shards. Restarting the launcher skips only shards whose completion
marker and payload hash both validate.

