# Stage 3 mechanism reverse audit

This is a code-to-result localization audit of the frozen NCER-AA implementation. It does not propose a new idea.

## Executed path

`stage3_collection._context_arrays` binds the observable context to the exact support branch initial state. `stage3_data.raw_context` concatenates current observable state/mask, two history deltas and masks, two previous actions and masks, current contact, nominal chunk and task ID. `stage3_models.create_biencoder` embeds the 256 residual bank; deterministic ID-stable FPS selects K=64. `create_pair_ranker` computes

`||dt-dc||₂ × softplus(MLP(h, mean(dt,dc), |dt-dc|, dt·dc))`,

which gives exact symmetry and exact zero self-distance by construction. `stage3_analysis.evaluate_records` carries frozen bank IDs through FPS/reranking and looks up the actually simulated candidate consequence; predictor scores never substitute for simulator outcomes.

## Controlled localization on development

Positive percentages mean C5 is better (lower error) than the named control.

| Information/mechanism | Frozen comparator | Oracle-regret improvement | Realized-effect improvement | Implementation isolation |
| --- | --- | --- | --- | --- |
| Nominal chunk | no_nominal_action | 0.741% | 0.662% | raw_context nominal slice → C4 symmetric scorer |
| Current state | state_shuffled_within_task | -3.606% | -3.207% | state+mask+contact bundle permutation |
| Short history | history_shuffled | -1.737% | -1.548% | two deltas/actions and masks permutation |
| Correct labels | consequence_labels_shuffled | 0.400% | 0.357% | true-distance row permutation |
| State + nominal jointly | joint_state_nominal_shuffled_within_task | 1.539% | 1.376% | joint within-task bundle permutation |
| All context | action_only_pair_ranker | 0.402% | 0.359% | constant context; target/candidate only |

Direct pair ranking versus the best vector regressor (C2_NC_TEMPORAL_VECTOR) changes oracle regret from 0.32655 to 0.31872 (2.399%). At realized execution, full-bank C4 error is 0.3589; K=64 C5 error is 0.37497. The difference between C4 and C5 localizes loss introduced by C3/FPS alphabet compression, not the pair scorer.

The dominant signed transition is C3→C4, not vector→ranker: C3 regret 0.24297, Spearman 0.66487, NDCG@16 0.61958, and realized error 0.28315; C4 changes these to 0.31872, 0.32664, 0.4449, and 0.3589. C3 alone improves realized error over B2 from 0.30817 to 0.28315, but the frozen C5 implementation discards C3 target-candidate distance after using C3 only to choose the 64-code atlas. C4 then reranks that atlas and reverses the C3 gain. Compression adds a smaller second loss: C4 0.3589 → C5 0.37497, while normalized utilization collapses from 0.24443 (C3) to 0.11399 (C5).

| Fresh support family | C3 regret | C4 regret | C3 NDCG@16 | C4 NDCG@16 |
| --- | --- | --- | --- | --- |
| smooth DCT (train overrepresented) | 0.25096 | 0.29881 | 0.61225 | 0.45152 |
| suffix-localized contact | 0.23985 | 0.36052 | 0.6244 | 0.43181 |
| low-rank temporal-action | 0.2381 | 0.29682 | 0.62208 | 0.45136 |

Training reuse supplied 48/24/24 branches per state for smooth/suffix/low-rank families, while fresh development is 32/32/32. The largest C3→C4 regret increase is on suffix-localized contact support. Because C3 sees the same training distribution and remains strong across all three families, imbalance can amplify but cannot by itself explain the cross-encoder failure.

The soft mixture changes K=64 realized error from 0.37497 (C5) to 0.37223 (C6), and oracle regret from 0.33479 to 0.33205. Its router sees only permitted observable context and uses no hard demonstration phase.

The pair scorer does not show mechanism-specific context dependence: C5 regret is 0.33479; nominal-shuffled=0.33416, state-shuffled=0.32314, history-shuffled=0.32908, and label-shuffled=0.33613. Several destructive controls are equal or better, so the absence of a positive Gate-B denominator—not a numerical division bug—is why gain retention is frozen to 1e9.

## What the code can and cannot establish

- Nominal conditioning can help because the same residual has different consequences under different base chunks; the nominal slice enters every proposed scorer before target/candidate comparison. A loss under no-nominal/shuffle controls supports this mechanism only if it is larger than run variance.
- Pair/listwise training optimizes candidate ordering directly, whereas C1/C2 first reconstruct a masked multi-group consequence vector and only then induce distances. A ranking gain therefore localizes avoidance of vector-reconstruction error accumulation.
- Here, that expected ranking advantage does not materialize consistently: C4 has slightly lower regret than the best vector regressor but worse NDCG/Recall, and it is much worse than the jointly trained C3 metric. All four calibration objective tuples produced poor C3-atlas+C4 regret, so the failure is not caused by one post-development objective choice.
- History can matter near contact because two observable deltas and previous actions distinguish approach, sustained contact and departure without using a future phase label. The history control permutes values and masks as one bundle.
- C3/FPS can lower performance when its learned embedding spreads candidates along directions irrelevant to the target-specific C4 scorer. C4 sees all 256 candidates; C5 sees only the 64 retained by C3, so C4→C5 degradation is the clean compression bottleneck.
- C6 can help only when the observable router separates regimes that need different pairwise scalings. If C6 or its shuffled-route control matches C5, the extra experts did not provide mechanism-specific routing.
- The controls have one member while primary ensembles have three, as preregistered. Large differences are informative, but small differences cannot be attributed solely to the ablated input because ensemble size is a remaining confound.
- C0 reproduces the prior hard-phase implementation and is used as a conservative comparator; it is not permitted as the proposed deployable method.

## Confirmation-boundary consequence

The holdout label is `FORCED_EXPLORATORY_HOLDOUT`. The pre-result replay incident means it is not untouched confirmation and cannot unlock BC, even if its counterfactual statistics pass.
