# R13-P15 Stage 1.5 Report — Failure Localization and Rescue Audit

## Executive result

Stage 1 remains rejected, and no Stage 1.5 revised deployable method passed the preregistered old-test internal screen. The stopping rule therefore prohibited collection of a fresh holdout. The exact Stage 1.5 disposition is given at the end of this report.

The nominal old-test gains of M2 and M5 are not evidence for consequence geometry: both reconstruct the globally repeated deterministic perturbation actions to floating-point precision, while the permuted-J and random-SPD controls retain 52%–60% of those gains. For M4 RECA, permuted-J retains 114.4% of the gain and random-SPD retains 99.98%.

This is a diagnostic-only experiment. No ACT, Diffusion Policy, SmolVLA, pi0.5, DINO-WM, behavior cloning, policy training or fresh-holdout collection was started.

## Evidence boundary and preregistration

- Preregistration/input-binding commit: `9a3ac1a4c774103fe618bd283909c2793ed581ec`.
- Frozen-method/old-test-plan commit: `aa82d46c5e0828956aef15918c2aa7656844472f`.
- `PREREGISTRATION.md` and `STAGE1_INPUT_BINDING.json` were committed before any revised-method result was computed or inspected.
- Primary K was 64. K=32 and K=128 were not inspected.
- Stage 1 test episodes 12–15 are retrospective/internal-screen evidence only, never confirmatory evidence for a revised method.
- Simulation ran locally on CPU with `CUDA_VISIBLE_DEVICES` empty, `MUJOCO_GL=glx`, renderer and offscreen renderer disabled. Four tasks ran in task-level CPU parallelism; no GPU was used.

## Frozen Stage 1 inputs and byte identity

Stage 1 remains `REJECT_CORE_HYPOTHESIS`; nothing in `experiments/r13_p15_caaa_v2/stage1/` was modified.

| input | value |
|-|-|
| repository input commit | 434427af0f8adc844851c27cfc050b2c9c6752dc |
| repository input tree | 3d3c93ab3903b5fc3be67c50ef16478be2de7503 |
| Stage 1 formal commit | 34995e8e7c3069b22785ad04536f0d429e75c0fc |
| Stage 1 formal Git tree | ad6fa59b782f63624ee3ccef8e880a2398669ce8 |
| LIBERO commit | 8f1084e3132a39270c3a13ebe37270a43ece2a01 |
| LIBERO source tree SHA-256 | e9197ca08fe4d7325f561fc40d7425167830253e0f0fceb1af2663b23292f71f |
| environment lock SHA-256 | f4421974cf948bfa765098e24819d445b209589611cbc3fe11e04c30fb0f0d3e |
| complete Stage 1 tree SHA-256 | 047aae35193339a460cd1dbac0e4495d7f9cff4a1cb2799c58b738e86e0e4c5c |
| branch rollout tree SHA-256 | 083893fd04a7e8282fc1ae0ba8ad2d362a87070c92f0bc780b672d8f818df59e |
| Jacobian metrics SHA-256 | 901f33cccf0378184c2d19303a4e1a433d7dfbab4e1abf9370371b3902bcc1ff |
| codebook tree SHA-256 | 56cbc8d020c8dcad14713acbd416ded890ac63783429907198bdd31e233187d3 |
| quantization JSONL SHA-256 | 9ac24b739273216cf2465652dc68cab8cb47956eabd131769a16336e14de2190 |
| quantized shard tree SHA-256 | 3385afd0b6e06567f2e2ad380c83e47937617ddbd8a39bc8cb859bc86d5d0167 |
| Stage 1 report SHA-256 | 5924ee1a77bc8c9339c5f450a8be87b9f8ad1e1a91b683fbf848f5f3f2047dd5 |

The pre-run Stage 1 release verifier passed all 256 replay tests, all 256 branch shards, 256 Jacobians, 128 quantization plans, 128 realized shards and all published hashes. The final repository verifier repeats the path-level Git identity check and bound-hash checks.

## Why Stage 1 remains rejected

The frozen Stage 1 CAAA result was 39.62% worse than covariance on pooled test error (frozen Stage 1 95% CI -168.41% to -5.50%). Stage 1 also showed severe action amplification, clipping and collapse. Stage 1.5 does not reinterpret that evidence.

## Retrospective failure localization

Across all 256 frozen states, the medians were:

| diagnostic | median |
|-|-|
| local_r2 | 0.711577 |
| local_normalized_rmse | 0.44687 |
| antithetic_nonlinearity_mean | 0.359868 |
| radius_derivative_drift_mean | 0.407164 |
| contact_mode_switch_rate | 0 |
| effective_rank | 1.790236 |
| condition_number | 5622.127188 |
| pseudoinverse_operator_norm | 0.659104 |
| selected_center_reachable_residual_mean | 0.491201 |
| realized_clipped_coordinate_fraction | 0.834201 |
| assignment_utilization | 0.015625 |

Per-phase medians:

| phase | R2 | NRMSE | antithetic | radius drift | eff. rank | condition | center residual | clip fraction |
|-|-|-|-|-|-|-|-|-|
| free_space | 0.940781 | 0.167223 | 0.024853 | 0.021094 | 2.989872 | 12.637382 | 0.996161 | 0.728733 |
| pre_contact | 0.565068 | 0.750442 | 0.54981 | 0.633639 | 1.61099 | 6671.075154 | 0.440571 | 0.875 |
| contact_onset | 0.618278 | 0.566386 | 0.40216 | 0.473244 | 1.470009 | 6430.127311 | 0.250173 | 0.875 |
| post_contact | 0.648797 | 0.662295 | 0.467769 | 0.599329 | 1.296961 | 6066.365479 | 0.160812 | 0.458333 |

Free-space Jacobians were locally accurate (median R2 0.941, NRMSE 0.167), but contact-onset, pre-contact and post-contact were substantially nonlinear. The median effective rank was 1.79 and condition number 5,622. M0 assigned only one of 64 codes at the median state (utilization 0.015625) and clipped 63.20% of continuous coordinates when pooled over realized old-test rows.

The frozen normalized metric was overwhelmingly driven by contact/force dimensions:

| consequence group | mean squared normalized error | share |
|-|-|-|
| object_pose | 1.020491e+07 | 2.098696e-05 |
| tcp_object_relative_pose | 1.158262e+05 | 2.382028e-07 |
| contact_and_force | 4.862272e+11 | 0.999953 |
| gripper_and_articulation | 1.288617e+05 | 2.650110e-07 |
| task_progress | 1.199279e+07 | 2.466381e-05 |
| constraint_violations | 5.916109e+05 | 1.216680e-06 |

The descriptive standardized regression (128 states with realized M0 rows, task and phase indicators; R2 0.722523) was:

| diagnostic term | standardized coefficient |
|-|-|
| local_model_error | 0.364753 |
| center_infeasibility | -0.350631 |
| inverse_amplification | 0.281383 |
| clipping | -0.380004 |
| codebook_collapse | -0.314685 |

These associations are descriptive, not causal. Signs are unstable under strong collinearity—for example clipping has a negative conditional coefficient even though M0 clipping is severe—so the mechanism interventions below receive more weight than this regression.

### Failure localization conclusion

The dominant M0 implementation failure was applying local perturbation geometry to uncentered full actions, followed by inverse amplification, clipping and one-code collapse. Centering/raw residual methods remove that execution pathology. However, CARA does not rescue CAAA, and RECA is reproduced by geometry-destroying controls. Local-model error is also material in contact phases: O1 gains 96.63% over M1 while O2 gains only 52.01%. Prototype infeasibility is not dominant (M4 token infeasibility 1.03%, clipping 0%). Thus the Stage 1 failure is not a single fixable clipping bug; the consequence-J geometry lacks mechanism specificity under this audit.

## Frozen methods and calibration

All deployable methods used the same Stage 1 train/calibration episodes, K=64, target actions, consequence schema/scales, action bounds, snapshots, simulator semantics and unchanged gripper commands.

- **M0 CAAA:** byte-frozen Stage 1 CAAA-v2 K=64 realized rows
- **M1 covariance:** byte-frozen Stage 1 covariance-Mahalanobis K=64 realized rows
- **M2 centered covariance:** global covariance-whitened K=64 k-means on train radius-0.10 residual actions; add state a0
- **M3 CARA:** global K=64 k-means on T_s delta_a; state pseudoinverse then add a0
- **M4 RECA:** global K=64 train realized-effect prototypes; bounded/radius-constrained local ridge decode
- **M5 phase residual:** four phase-specific K=64 raw residual-action codebooks
- **M6 permuted-J RECA:** M4 with Jacobians cyclically permuted within task/split/phase
- **M7 random-SPD:** matched-spectrum deterministic random transform with the M4 constrained decoder
- **O1 true-effect oracle:** same-state radius-0.05 dictionary selected by true settled consequence
- **O2 linear-J oracle:** same dictionary selected by frozen local linear J delta_a

RECA calibration selected beta=1.000000e-06, residual radius cap=0.1, feasibility quantile=0.99 and threshold=3.628425e+06. Selection used only episodes 8–11. The realized old-test execution comprised 64 states and 18,432 revised-method branches; every completion marker, plan binding and finite-value check passed.

## Old-test internal-screen results

Pooled results (all intervals below remain retrospective):

| method | settled error | rel. gain vs M1 | paired 95% CI | action error | clip | contact | progress | infeasible |
|-|-|-|-|-|-|-|-|-|
| M0 CAAA | 1.028925e+05 | -0.396227 | [-1.703449, -0.057026] | 0.774718 | 0.632039 | 0.729167 | 0.829102 | 0 |
| M1 covariance | 7.369326e+04 | 0 | [0, 0] | 0.094905 | 0.000556 | 0.808594 | 0.943034 | 0 |
| M2 centered covariance | 2211.808248 | 0.969986 | [0.857483, 0.999491] | 5.356678e-17 | 0 | 1 | 1 | 0 |
| M3 CARA | 1.652496e+05 | -1.242397 | [-5.102599, 0.218897] | 0.102233 | 0.064494 | 0.954753 | 0.957682 | 0 |
| M4 RECA | 3.620555e+04 | 0.508699 | [-0.169146, 0.81281] | 0.028605 | 0 | 0.950846 | 0.983724 | 0.010254 |
| M5 phase residual | 1613.889795 | 0.9781 | [0.899894, 0.999115] | 1.074340e-17 | 0 | 0.999674 | 1 | 0 |
| M6 permuted-J RECA | 3.081796e+04 | 0.581808 | [0.126529, 0.793678] | 0.028605 | 0 | 0.93457 | 0.987305 | 0.010254 |
| M7 random-SPD | 3.621143e+04 | 0.50862 | [-0.007493, 0.766639] | 0.023752 | 0 | 0.963867 | 0.98763 | 0.009277 |
| O1 true-effect oracle | 2479.844441 | 0.966349 | [0.921519, 0.981038] | 0.018361 | 0 | 0.96582 | 1 | 0 |
| O2 linear-J oracle | 3.536523e+04 | 0.520102 | [-0.057249, 0.788377] | 0.01745 | 0 | 0.965169 | 0.98763 | 0 |

Per-task settled error:

| task | M0 CAAA | M1 covariance | M2 centered covariance | M3 CARA | M4 RECA | M5 phase residual | M6 permuted-J RECA | M7 random-SPD | O1 true-effect oracle | O2 linear-J oracle |
|-|-|-|-|-|-|-|-|-|-|-|
| bowl_on_plate | 1.528411e+05 | 9.284181e+04 | 1045.28667 | 2.977269e+05 | 8.200461e+04 | 1147.048516 | 7.389450e+04 | 7.549611e+04 | 6780.356227 | 7.157669e+04 |
| plate_push | 1143.214419 | 429.8537 | 0.041732 | 72.345852 | 77.713 | 0.060672 | 80.027537 | 73.673909 | 33.287855 | 36.526012 |
| stove_turn_on | 2790.866073 | 522.277257 | 2.516133 | 105.577338 | 99.332838 | 2.48355 | 109.521939 | 105.489157 | 25.316331 | 57.886402 |
| wine_rack | 2.547950e+05 | 2.009791e+05 | 7799.388456 | 3.630934e+05 | 6.264056e+04 | 5305.966441 | 4.918780e+04 | 6.917044e+04 | 3080.417352 | 6.978982e+04 |

Pooled-across-task per-phase settled error:

| phase | M0 CAAA | M1 covariance | M2 centered covariance | M3 CARA | M4 RECA | M5 phase residual | M6 permuted-J RECA | M7 random-SPD | O1 true-effect oracle | O2 linear-J oracle |
|-|-|-|-|-|-|-|-|-|-|-|
| free_space | 267.306802 | 49.992957 | 5.099549e-13 | 11.650895 | 11.459051 | 1.186526e-13 | 11.474717 | 17.100044 | 5.342683 | 5.343022 |
| pre_contact | 3720.052154 | 454.292631 | 1.103496 | 146.533484 | 166.650293 | 1.066802 | 160.302405 | 144.414159 | 41.02912 | 91.843791 |
| contact_onset | 5292.862732 | 446.702604 | 2.926948 | 139.268095 | 153.39805 | 2.749734 | 197.67267 | 160.500461 | 49.667036 | 87.634399 |
| post_contact | 4.022900e+05 | 2.938221e+05 | 8843.202547 | 6.607008e+05 | 1.444907e+05 | 6451.742643 | 1.229024e+05 | 1.445237e+05 | 9823.338925 | 1.412761e+05 |

Candidate screen gates:

| candidate | pooled gain | tasks | clip reduction | action degradation | M6 retention | M7 retention | pass |
|-|-|-|-|-|-|-|-|
| M2 centered covariance | 0.969986 | 4 | 1 | -1 | 0.59981 | 0.524357 | False |
| M3 CARA | -1.242397 | 2 | 0.897959 | 0.077216 | NA | NA | False |
| M4 RECA | 0.508699 | 4 | 1 | -0.698596 | 1.143716 | 0.999843 | False |
| M5 phase residual | 0.9781 | 4 | 1 | -1 | 0.594835 | 0.520008 | False |

No candidate passed. M2/M5 exploit the fact that the same 48 signed radius-0.10 perturbations appear in train and old test: their action errors are approximately 5.36e-17 and 1.07e-17. The remaining nonzero effect error despite nearly identical actions is concentrated in highly scaled contact/force channels, exposing numerical/contact sensitivity of this retrospective design rather than action-alphabet generalization.

## Pooled and per-task confidence intervals

All 10,000-replicate intervals use paired episode clusters resampled within task. They characterize the old-test internal screen only.

| method | pooled | bowl_on_plate | plate_push | stove_turn_on | wine_rack |
|-|-|-|-|-|-|
| M0 CAAA | -0.396227 [-1.703449, -0.057026] | -0.646253 [-1.920779, -0.22626] | -1.659543 [-4.503847, -0.699256] | -4.343648 [-10.282195, -0.447655] | -0.267768 [-2.944359, 0.074755] |
| M2 centered covariance | 0.969986 [0.857483, 0.999491] | 0.988741 [0.980837, 0.998769] | 0.999903 [0.999615, 1] | 0.995182 [0.989678, 1] | 0.961193 [0.657943, 1] |
| M3 CARA | -1.242397 [-5.102599, 0.218897] | -2.206819 [-7.271094, -1.089696] | 0.831697 [0.699924, 0.876185] | 0.797852 [0.684566, 0.839513] | -0.806623 [-5.878061, 0.839888] |
| M4 RECA | 0.508699 [-0.169146, 0.81281] | 0.116728 [-0.42053, 0.710535] | 0.819211 [0.683604, 0.866298] | 0.809808 [0.671302, 0.870362] | 0.688323 [-0.033907, 0.889061] |
| M5 phase residual | 0.9781 [0.899894, 0.999115] | 0.987645 [0.980837, 0.996704] | 0.999859 [0.999441, 1] | 0.995245 [0.989811, 1] | 0.973599 [0.767298, 1] |
| M6 permuted-J RECA | 0.581808 [0.126529, 0.793678] | 0.204082 [-0.054289, 0.521005] | 0.813826 [0.669511, 0.862549] | 0.790299 [0.653821, 0.841589] | 0.755259 [0.235211, 0.877359] |
| M7 random-SPD | 0.50862 [-0.007493, 0.766639] | 0.186831 [0.017027, 0.411381] | 0.828607 [0.694287, 0.873956] | 0.798021 [0.635867, 0.862957] | 0.655833 [-0.221, 0.878603] |
| O1 true-effect oracle | 0.966349 [0.921519, 0.981038] | 0.926969 [0.836548, 0.965176] | 0.92256 [0.8637, 0.942432] | 0.951527 [0.915662, 0.968323] | 0.984673 [0.967012, 0.988995] |
| O2 linear-J oracle | 0.520102 [-0.057249, 0.788377] | 0.229047 [-0.151924, 0.660063] | 0.915027 [0.855048, 0.935277] | 0.889165 [0.798833, 0.915137] | 0.652751 [-0.141838, 0.868558] |

## Mechanism and oracle gaps

- O1 versus M1: 96.63% lower error, establishing a local dictionary upper bound on this old perturbation support.
- O1 versus O2: O1 error 2479.844441 versus O2 3.536523e+04; the 44.62 percentage-point gain gap identifies substantial local-model loss.
- O2 versus M4: O2 error 3.536523e+04 versus M4 3.620555e+04, leaving a small global-prototype/decoder gap relative to the much larger model gap.
- M3 versus M0: CARA reduces pooled clipping from 0.632039 to 0.064494 but increases settled error from 1.028925e+05 to 1.652496e+05. Centering fixes clipping but not CAAA geometry.
- M4 versus M6: permuted-J is better (3.081796e+04 versus 3.620555e+04) and retains 114.37% of the M4 gain.
- M4 versus M7: random-SPD is effectively identical (3.621143e+04 versus 3.620555e+04) and retains 99.98% of the M4 gain.

## Fresh holdout

Fresh IDs: none. Preferred IDs 16–23 were never read, validated, selected or executed because the Part E internal screen failed. `fresh_holdout_split.json` and `fresh_branch_rollouts.zarr` both carry status `NOT_COLLECTED_INTERNAL_SCREEN_FAILED` with zero records/states. This is a stopping manifest, not confirmatory data.

## Failed and negative runs

- `stage1-5-prepare-old`: INTERRUPTED_IMPLEMENTATION_PERFORMANCE_FAILURE (exit 130). The first constrained-decoder implementation redundantly recomputed the same matrix spectral norm and normal equations for every prototype at a state. It was interrupted before method definitions, codebooks, plans, or revised-method result summaries were written.
- `stage1-5-prepare-old`: READY_FOR_OLD_TEST_INTERNAL_SCREEN_COLLECTION (exit 0). Completed as specified.
- `old-test shard validation`: VALIDATOR_IMPLEMENTATION_FAILURE (exit 1). The first ad-hoc validator passed equal_nan=True to NumPy array_equal for string arrays, which raises TypeError. No payload was changed.
- `stage1-5-screen-old`: SUPERSEDED_NON_STRICT_JSON (exit 0). The first successful screen encoded undefined gain retention for the negative-gain M3 candidate as Python Infinity. Although Python could read it, Infinity is not strict JSON.
- `stage1-5-screen-old`: REJECT_P15_FAMILY_STRICT_JSON (exit 0). No revised deployable method passed all Part E gates; fresh holdout collection was prohibited.
- Simulator collection: 64/64 shards and 18,432/18,432 revised branches completed; zero marker, plan-binding or finite-value failures.
- M3 CARA was negative: 124.24% worse than M1 and 60.61% worse than M0 despite reduced clipping.
- M2, M4 and M5 were rejected at the mechanism-specificity gate; M6/M7 reproduced too much or all of their gains.
- Fresh-holdout collection was intentionally not run; this is compliance with the stopping rule, not missing execution.

## Artifact hashes

| artifact | SHA-256 |
|-|-|
| PREREGISTRATION.md | 6f13cdeceda9c782f63e20fcea3d085db33b1adabf5a4d0b945b54db7d88b135 |
| STAGE1_INPUT_BINDING.json | ed5d4e87b75442c0d6efbaa9232552ee01b9c959a94f2df110fa849c0acf5810 |
| retrospective_diagnostics.parquet | 04aa9f6a47284e83d0cd033360fe72d2b5f7719f4a5c12a9ae78c1da64f85a15 |
| error_decomposition.csv | 64175a1e7ead5c6ca3a1eef1f216671d9b569cce127dc6cf8fb72cc0e25fab65 |
| fresh_holdout_split.json | 19211f3dd56cae72311982c7bafd7ae1a72f6a6acfdd4aa39ddc501702884c47 |
| fresh_branch_rollouts.zarr | f1741bf70e6a114859fa2455ef41b6464afd6320cac4e3ba7e8607d910354f9c |
| method_definitions.json | a8c659eb1fa1a48ddbd715c1b2749048b24abd5867546c3f12f985b16c5e38ec |
| quantization_results_by_task.csv | 26c0b895be9675b2ae5b4c89ca6797789e88835713ac33002b39d48ea47f27b7 |
| quantization_results_by_phase.csv | 20e7654bb903361d166a07d122c8aa95dec7d43b2dd4eb147020b599ff57f8d0 |
| mechanism_controls.csv | f5284680c1476bc5701c883ccacbb2f64022d568ff18f6dd9b049e866436799b |
| bootstrap_results.json | f9e0f24b9559c74d2b82e1889e6ebda752407a40725cc17f915647d5ad2aeba6 |

## Next permitted experiment

Do not start policy training. If this line of inquiry is revisited, preregister a new evaluation with held-out perturbation directions or naturally varying demonstration residuals so train and test action supports are not identical; use an empirical nonlinear local-effect dictionary or contact-mode-stratified model, and add a non-force-dominated robustness metric only as a separately preregistered secondary analysis. That would be a new experiment, not Stage 1.5 continuation.

## Final disposition

FINAL_DISPOSITION: REJECT_P15_FAMILY
