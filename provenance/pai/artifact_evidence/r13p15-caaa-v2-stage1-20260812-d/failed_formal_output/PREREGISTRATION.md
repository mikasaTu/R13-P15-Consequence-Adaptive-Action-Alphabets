# R13-P15-v2 Stage 1 Preregistration — LIBERO adaptation

## Question

Does a local physical-consequence pullback metric organize four-step action
chunks better than Euclidean geometry, action covariance, global and
phase-conditioned k-means, PCA+k-means, and the prior diagonal-sensitivity
implementation?

## Authorized scope change

The source plan specified ManiSkill3 and prohibited PAI. The latest user
instruction supersedes those two platform constraints: this run uses standard
LIBERO and executes formal collection/evaluation on PAI after a bounded local
CPU or one-GPU smoke. It still trains no policy and stops after Stage 1.

LIBERO has no `state_dict` API. The equivalent privileged oracle is the
MuJoCo simulator state (`qpos`, `qvel`, body/site poses, joint coordinates and
contacts). It is used only to construct physical-consequence labels and never
as policy input.

## Frozen task set

All tasks are from `libero_goal` and use the official successful
demonstrations:

1. `put_the_bowl_on_the_plate` — low-constraint pick/place control;
2. `push_the_plate_to_the_front_of_the_stove` — sustained sliding contact;
3. `turn_on_the_stove` — articulated, contact-sensitive switch motion;
4. `put_the_wine_bottle_on_the_rack` — precision oriented placement.

All methods use Panda `OSC_POSE`, 20 Hz, and exactly the same normalized 7D
action semantics. The alphabet acts on the 24 continuous coordinates formed
by four steps × six delta-pose channels. The four gripper commands are copied
unchanged from the demonstration for perturbation, encoding and decoding.

## Episodes, phases and split

The first 16 official successful demonstrations for each task are frozen.
Episode IDs 0–7 are train, 8–11 calibration, and 12–15 test. Four snapshots
are chosen per episode: free-space, pre-contact, contact onset, and
post-contact. Selection uses only recorded state/contact/progress and a frozen
deterministic rule; fallbacks and saturation-driven index shifts are logged.

## Branch design

At every snapshot, `H=4`. A seeded QR decomposition gives 24 deterministic
orthonormal directions in the 24D continuous chunk. Each direction is run at
signs ± and normalized radii 0.05 and 0.10. Pose actions must remain in
`[-1,1]`; a phase index is shifted to the nearest semantically valid
unsaturated chunk rather than silently changing radius. Every branch starts
from the identical restored MuJoCo state and restored controller state.

Immediate consequences are captured after four actions. Settled consequences
are captured after three zero-delta-pose controls while holding the last
gripper command fixed.

## Consequence metric

The frozen schema contains masked, robustly scaled continuous features for:
primary-object pose; TCP pose; TCP-to-primary pose; primary-to-target pose;
gripper width; articulated joint coordinates; task progress; task-relevant
contact impulse; and constraint violation. Rotation uses the continuous 6D
representation formed by the first two rotation-matrix columns.
Success and the categorical contact transition are evaluation outcomes, not
continuous Jacobian dimensions. Robust centers/scales use train episodes only.

For each snapshot, local ridge Jacobians are fit from antithetic branches.
Ridge, singular cutoff, PCA rank and metric regularization are selected using
calibration episodes only. Test episodes never select a hyperparameter or the
strongest comparison baseline.

## Compared methods

- Euclidean farthest-point medoids;
- covariance Mahalanobis;
- global k-means;
- phase-conditioned k-means;
- PCA+k-means;
- old diagonal sensitivity;
- random SPD with a matched spectrum;
- state-permuted Jacobian;
- full CAAA-v2.

`K=64` is primary. `K=32` and `K=128` are sensitivity analyses. The primary
comparison gives every method the same true-CAAA null-space continuous
residual, thereby conservatively isolating tokenized sensitive-space geometry.
Native-null results for the two mechanism controls are diagnostic only.

## Primary outcomes and bootstrap

The primary outcome is settled realized physical-effect quantization error:
each decoded action is re-executed from the snapshot. Secondary outcomes are
immediate error, metric-to-consequence Spearman correlation, contact-mode and
task-progress preservation, action reconstruction, codebook utilization,
local linearity, effective rank and condition number. Confidence intervals
use 10,000 deterministic episode-clustered bootstrap replicates.

The strongest non-consequence baseline is selected on calibration episodes
only and frozen before test aggregation.

## Disposition gates

`GO_TO_SMALL_BC` requires all of:

1. at least 10% pooled test effect-error reduction versus the frozen strongest
   non-consequence baseline;
2. episode-clustered 95% CI supporting positive improvement;
3. improvement in at least two of plate-push, stove-turn-on and wine-rack;
4. bowl-on-plate degradation no worse than 5%;
5. permuted-J retains at most 25% of the CAAA gain;
6. random-SPD does not reproduce the main gain;
7. action reconstruction degrades by no more than 10%;
8. dead-code ratio below 20%.

`REJECT_CORE_HYPOTHESIS` is returned if improvement is below 5%, k-means is
stronger, permuted-J retains over 50% of gain, random-SPD is comparable, or
the oracle metric is unstable on test episodes. All other non-passing cases
return `REVISE_ALPHABET`.

Exactly one disposition is written. No ACT, Diffusion Policy, SmolVLA, π0.5,
DINO-WM or other policy/world-model training starts automatically.
