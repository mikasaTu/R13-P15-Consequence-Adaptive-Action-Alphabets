"""Command-line entry points for local smoke and resumable PAI Stage 1."""

from __future__ import annotations

import argparse
import json
import os

from . import config


def _paths(args):
    paths = config.resolved_paths(args)
    paths["output_root"] = args.output_root or paths["output_root"]
    return paths


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def formal_setup(paths, output_root):
    from .pipeline import freeze_environment, scaffold, select_snapshots, validate_replay
    from .storage import atomic_json

    scaffold(output_root, _project_root())
    lock = freeze_environment(paths, _project_root(), output_root, hash_demos=True)
    split = select_snapshots(paths, output_root)
    replay = validate_replay(
        paths,
        output_root,
        episode_limit=config.N_EPISODES,
        tolerance=1e-12,
    )
    incident = {
        "scope": "local_development_smoke_before_formal_submission",
        "failed_test": {
            "task_id": "bowl_on_plate",
            "episode_id": 0,
            "phase": "free_space",
            "final_state_max_abs": 0.012277476686433508,
            "immediate_consequence_max_abs": 0.01192543045445929,
            "settled_consequence_max_abs": 0.0022943595780261228,
        },
        "root_cause": (
            "Panda gripper.current_action is integrated across MuJoCo substeps and is omitted "
            "from flattened simulator state"
        ),
        "fix": "restore gripper action history plus MuJoCo control and solver auxiliary arrays",
        "post_fix_formal_gate": replay["gate"],
    }
    atomic_json(os.path.join(output_root, "work", "development_replay_incident.json"), incident)
    if not replay["passed"]:
        raise RuntimeError("formal replay validation failed")
    return {
        "environment_lock": os.path.join(output_root, "environment_lock.json"),
        "snapshots": len(split["records"]),
        "replay_tests": replay["n_tests"],
        "replay_gate": replay["gate"],
        "demo_count": len(lock["demonstrations"]),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "smoke",
        "formal-setup",
        "collect-branches",
        "prepare-analysis",
        "collect-quantized",
        "finalize",
        "stage1-5-prepare-old",
        "stage1-5-collect-old",
        "stage1-5-screen-old",
    ))
    parser.add_argument("--libero-source", default=None)
    parser.add_argument("--libero-env", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--stage1-root", default=None)
    parser.add_argument("--task-id", choices=[task["task_id"] for task in config.TASKS])
    parser.add_argument("--plan-limit", type=int, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = _paths(args)
    output_root = paths["output_root"]
    if args.command == "smoke":
        from .pipeline import smoke

        result = smoke(paths, _project_root(), output_root)
    elif args.command == "formal-setup":
        result = formal_setup(paths, output_root)
    elif args.command == "collect-branches":
        from .pipeline import collect_branches

        result = {
            "shards": len(
                collect_branches(
                    paths,
                    output_root,
                    task_ids=[args.task_id] if args.task_id else None,
                )
            )
        }
    elif args.command == "prepare-analysis":
        from .analysis import prepare_analysis

        result = prepare_analysis(output_root)
    elif args.command == "collect-quantized":
        from .quantization import collect_quantized

        result = {
            "shards": len(
                collect_quantized(
                    paths,
                    output_root,
                    task_id=args.task_id,
                    plan_limit=args.plan_limit,
                )
            )
        }
    elif args.command == "finalize":
        from .reporting import finalize

        result = finalize(output_root)
    elif args.command.startswith("stage1-5-"):
        from .stage1_5 import collect_old_test, prepare_old_test, screen_old_test

        stage1_root = args.stage1_root or os.path.join(
            _project_root(), "experiments", "r13_p15_caaa_v2", "stage1"
        )
        stage1_5_root = args.output_root or os.path.join(
            _project_root(), "experiments", "r13_p15_caaa_v2", "stage1_5"
        )
        if args.command == "stage1-5-prepare-old":
            result = prepare_old_test(stage1_root, stage1_5_root)
        elif args.command == "stage1-5-collect-old":
            result = {
                "shards": len(
                    collect_old_test(
                        paths,
                        stage1_5_root,
                        task_id=args.task_id,
                        plan_limit=args.plan_limit,
                    )
                )
            }
        elif args.command == "stage1-5-screen-old":
            result = screen_old_test(
                stage1_root,
                stage1_5_root,
                replicates=args.bootstrap_replicates,
            )
        else:
            raise AssertionError(args.command)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
