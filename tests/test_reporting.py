from caaa_libero import config
from caaa_libero.analysis import BASELINE_METHODS
from caaa_libero.reporting import choose_baseline


def test_choose_baseline_is_calibration_only_and_excludes_consequence_methods():
    rows = []
    for index, method in enumerate(BASELINE_METHODS):
        rows.append(
            {
                "split": "calibration",
                "k": config.PRIMARY_K,
                "method": method,
                "settled_effect_error": float(index + 1),
            }
        )
    rows.extend(
        [
            {
                "split": "calibration",
                "k": config.PRIMARY_K,
                "method": "caaa_v2",
                "settled_effect_error": 0.0,
            },
            {
                "split": "test",
                "k": config.PRIMARY_K,
                "method": BASELINE_METHODS[-1],
                "settled_effect_error": -100.0,
            },
        ]
    )

    selected, means = choose_baseline(rows)

    assert selected == BASELINE_METHODS[0]
    assert set(means) == set(BASELINE_METHODS)
    assert "caaa_v2" not in means
