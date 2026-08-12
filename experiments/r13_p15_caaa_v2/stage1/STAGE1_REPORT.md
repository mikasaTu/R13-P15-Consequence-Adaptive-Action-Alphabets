# R13-P15-v2 Stage 1 Report — LIBERO CAAA-v2

## Executive result

**REJECT_CORE_HYPOTHESIS**

The frozen calibration baseline was `covariance_mahalanobis`. The pooled test relative improvement of CAAA-v2 was
-0.39623, with episode-clustered 95% CI [-1.6841, -0.055]. This is a mechanism-only oracle audit: no ACT,
Diffusion Policy, SmolVLA, π0.5, DINO-WM, behavior cloning, or other policy training was launched.

## Scope and frozen environment

This authorized LIBERO adaptation uses standard `libero_goal`, Panda `OSC_POSE`, 20 Hz, 7D normalized actions,
H=4, and an alphabet over the 24 pose coordinates while copying gripper commands unchanged. It freezes 16
successful official demonstrations per task with episode split 8/4/4 and four snapshots per episode.

- Project commit: `34995e8e7c3069b22785ad04536f0d429e75c0fc`
- Project tree SHA-256: `df66a9429fe2a36cbca2947b0bbdf7e1dfee80f514a0160ce22be986ea0ff3da`
- LIBERO upstream commit: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- LIBERO source tree SHA-256: `e9197ca08fe4d7325f561fc40d7425167830253e0f0fceb1af2663b23292f71f`
- Python: `3.8.13 | packaged by conda-forge | (default, Mar 25 2022, 06:04:18) `
- MuJoCo: `2.3.7`; robosuite: `1.4.0`; PyTorch: `1.11.0+cu113`; CUDA build: `11.3`
- Formal demonstration SHA-256 values are recorded individually in `environment_lock.json`.

## Replay validation and all failures

Formal replay gate: **PASS** (256 tests, 0 failures, tolerance 1e-12).
The formal failed-test array contains: `[]`.

Development incident retained for completeness: The first local smoke initially failed for bowl_on_plate/e0/free_space: repeated A had final-state max |Δ|=0.0122775, immediate consequence max |Δ|=0.0119254, and settled max |Δ|=0.00229436. Cause: Panda gripper.current_action is an integrated hidden command omitted by MuJoCo's flattened state. Snapshotting gripper history plus solver/control auxiliaries reduced all formal A/A and A/B/A differences to zero.

## Frozen consequence model and calibration

The task-generic continuous schema has 46 masked dimensions: object/TCP/relative poses in continuous rotation-6D,
gripper width, articulation, task progress, three task-relevant contact-force channels, penetration and joint-limit
violation. Immediate effects are measured after H=4; settled effects add three zero-pose steps holding the final
gripper command. Train-only robust scales were used. Calibration-only selections were ridge=0.001, singular
cutoff=0.0001, metric regularization=1e-08, covariance regularization=1e-08, PCA rank=12.

### Per-task and per-phase locality (median across episodes)

| task | phase | R² | NRMSE | eff. rank | condition | CAAA Spearman |
| --- | --- | --- | --- | --- | --- | --- |
| bowl_on_plate | contact_onset | 0.53805 | 0.68639 | 1.5793 | 6621.1 | 0.53454 |
| bowl_on_plate | free_space | 0.93868 | 0.16899 | 3.5704 | 4233.5 | 0.99696 |
| bowl_on_plate | post_contact | 0.023927 | 1.327 | 1 | 6431.9 | 0.046926 |
| bowl_on_plate | pre_contact | 0.56888 | 0.60686 | 1.9374 | 7633 | 0.53541 |
| plate_push | contact_onset | 0.88463 | 0.21972 | 1.4945 | 8058.5 | 0.9673 |
| plate_push | free_space | 0.89339 | 0.17814 | 3.5351 | 4151.8 | 0.99555 |
| plate_push | post_contact | 0.67106 | 0.36866 | 3.0148 | 7665.7 | 0.88964 |
| plate_push | pre_contact | 0.74389 | 0.35112 | 1.6225 | 6669.1 | 0.8627 |
| stove_turn_on | contact_onset | 0.61869 | 0.68906 | 1.2622 | 1873.4 | 0.56919 |
| stove_turn_on | free_space | 0.94045 | 0.16545 | 2.0903 | 11.715 | 0.99718 |
| stove_turn_on | post_contact | 0.76405 | 0.48518 | 1.0853 | 2513.7 | 0.79557 |
| stove_turn_on | pre_contact | 0.5797 | 0.86752 | 1.6343 | 4526.7 | 0.47274 |
| wine_rack | contact_onset | 0.46319 | 0.6322 | 1.1909 | 6718.1 | 0.67576 |
| wine_rack | free_space | 0.9718 | 0.16713 | 2.3774 | 10.739 | 0.99815 |
| wine_rack | post_contact | 0.40933 | 0.974 | 1.2571 | 7734.1 | 0.34217 |
| wine_rack | pre_contact | 0.11951 | 0.81056 | 1.309 | 7219.3 | 0.3741 |

## Realized held-out quantization results (K=64)

Errors below come from executing every decoded action from its identical restored simulator snapshot. Progress
preservation means settled progress differs by at most 0.05. Codebook utilization is measured on test assignments.

| task | method | settled err | immediate err | contact | progress | action err | util. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bowl_on_plate | caaa_v2 | 1.5284e+05 | 2.3872e+05 | 0.81771 | 0.79427 | 0.79904 | 0.14062 |
| bowl_on_plate | covariance_mahalanobis | 92842 | 45572 | 0.84375 | 0.97005 | 0.099601 | 0.21875 |
| bowl_on_plate | old_diagonal_sensitivity | 91730 | 72135 | 0.89583 | 0.97266 | 0.48371 | 0.046875 |
| bowl_on_plate | permuted_j | 5.8696e+05 | 6.206e+05 | 0.88021 | 0.57682 | 0.96168 | 0.21875 |
| bowl_on_plate | random_spd | 82950 | 41111 | 0.84505 | 0.95964 | 0.37341 | 0.14062 |
| plate_push | caaa_v2 | 1143.2 | 878.41 | 0.6888 | 1 | 1.0665 | 0.015625 |
| plate_push | covariance_mahalanobis | 429.85 | 354.61 | 0.71094 | 1 | 0.14155 | 0.25 |
| plate_push | old_diagonal_sensitivity | 1224.6 | 1068.9 | 0.51823 | 1 | 0.67696 | 0.015625 |
| plate_push | permuted_j | 1305.2 | 1151.6 | 0.4388 | 1 | 1.0949 | 0.015625 |
| plate_push | random_spd | 1005.7 | 827.52 | 0.44661 | 1 | 0.54387 | 0.015625 |
| stove_turn_on | caaa_v2 | 2790.9 | 916.19 | 0.63542 | 0.6862 | 0.6828 | 0.015625 |
| stove_turn_on | covariance_mahalanobis | 522.28 | 352.83 | 0.84115 | 0.84635 | 0.07793 | 0.23438 |
| stove_turn_on | old_diagonal_sensitivity | 1600.8 | 1236.5 | 0.70833 | 0.7513 | 0.53893 | 0.015625 |
| stove_turn_on | permuted_j | 4902.7 | 2557.6 | 0.52734 | 0.5 | 1.0264 | 0.015625 |
| stove_turn_on | random_spd | 3544.3 | 1190.8 | 0.71484 | 0.4388 | 1.0096 | 0.015625 |
| wine_rack | caaa_v2 | 2.5479e+05 | 2.9017e+05 | 0.77474 | 0.83594 | 0.55055 | 0.125 |
| wine_rack | covariance_mahalanobis | 2.0098e+05 | 3.5655e+05 | 0.83854 | 0.95573 | 0.060532 | 0.14062 |
| wine_rack | old_diagonal_sensitivity | 1.6877e+05 | 6.2531e+05 | 0.69792 | 0.9401 | 0.36399 | 0.0625 |
| wine_rack | permuted_j | 5.4062e+05 | 4.7803e+05 | 0.66536 | 0.91797 | 1.0045 | 0.17188 |
| wine_rack | random_spd | 3.2874e+05 | 3.3726e+05 | 0.7513 | 0.89844 | 0.39739 | 0.29688 |

Full per-task results, including K=32/128 sensitivity, are in `results_by_task.csv`; per-phase results are in
`results_by_phase.csv`. All nine methods' metric-to-consequence Spearman correlations, local linearity, effective
rank and condition number are in `jacobian_metrics.parquet`.

## Episode-clustered confidence intervals

| scope | relative improvement | 95% CI |
| --- | --- | --- |
| pooled | -0.39623 | [-1.6841, -0.055] |
| bowl_on_plate | -0.64625 | [-1.9208, -0.22626] |
| plate_push | -1.6595 | [-4.5038, -0.69926] |
| stove_turn_on | -4.3436 | [-10.282, -0.44766] |
| wine_rack | -0.26777 | [-2.9444, 0.074755] |

Bootstrap uses 10,000 paired episode-cluster resamples within task. Calibration episodes selected the baseline;
test episodes were not used for model or baseline selection.

## Mechanism controls and disposition logic

The permuted-J and random-SPD realized controls are recorded in `mechanism_controls.csv`. The final gate also checks
improvement on contact-sensitive tasks, bowl-control degradation, action reconstruction, dead-code ratio, local
stability, and whether k-means or geometry-destroying controls reproduce the gain. Applying the preregistered gates
returns exactly:

**REJECT_CORE_HYPOTHESIS**

## Next recommended experiment

Do not start policy training for CAAA-v2. Test a narrower diagnostic that separates local linear-model failure from state-dependent codebook alignment, using the same frozen replay snapshots.

## Artifact hashes

| artifact | SHA-256 |
| --- | --- |
| PREREGISTRATION.md | 07e0ac3123dea4c51b18d41c5a8f989e8a18fb7b61a028e08e977a261f274856 |
| environment_lock.json | f4421974cf948bfa765098e24819d445b209589611cbc3fe11e04c30fb0f0d3e |
| task_and_seed_split.json | 7b11ac5dc44877d0b5011c355d178dda44558b35d33e4af7f705fba1bfe1cc22 |
| branch_replay_validation.json | 7ae428605a824ba7f36786ccffa7ef32e7a7bf795fa8f866b29d099baacb5cf9 |
| consequence_schema.json | 0d6545ef9917a2cd25f0016547a67157c674628d9cae79bca0b8a6fde66fced1 |
| branch_rollouts.zarr | 083893fd04a7e8282fc1ae0ba8ad2d362a87070c92f0bc780b672d8f818df59e |
| jacobian_metrics.parquet | 901f33cccf0378184c2d19303a4e1a433d7dfbab4e1abf9370371b3902bcc1ff |
| alphabet_codebooks | 56cbc8d020c8dcad14713acbd416ded890ac63783429907198bdd31e233187d3 |
| results_by_task.csv | 7b4a190e03d4134463d10e64870d157217f62a9210587c88330d0c747ee4953b |
| results_by_phase.csv | 78fc2d836ff83223e5f32341cedf307ab5e60f55f439c105b8a99a27d1ff9ad9 |
| bootstrap_results.json | 919fc0dcfd3fba71cc64fa7c8ba07d8ef013b14742322b137feff320899fb9fb |
| mechanism_controls.csv | e505cd93d56881d97be5bd535c3fa8e702dc91bbead20dea7b65ef232adbbba6 |

FINAL_DISPOSITION: REJECT_CORE_HYPOTHESIS
