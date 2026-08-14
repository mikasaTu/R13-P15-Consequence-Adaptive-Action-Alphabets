"""Command-line entry points for the frozen LIBERO experiments."""

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
        "stage1-5-finalize",
        "stage2-freeze-task",
        "stage2-finalize-freeze",
        "stage2-collect-support",
        "stage2-collect-candidates",
        "stage2-export-rollouts",
        "stage2-run-development",
        "stage2-refresh-summaries",
        "stage2-verify",
        "stage3-freeze",
        "stage3-refresh-freeze",
        "stage3-collect",
        "stage3-verify-training-reuse",
    ))
    parser.add_argument("--libero-source", default=None)
    parser.add_argument("--libero-env", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--stage1-root", default=None)
    parser.add_argument("--task-id", choices=[task["task_id"] for task in config.TASKS])
    parser.add_argument("--plan-limit", type=int, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument(
        "--splits",
        default=None,
        help="Comma-separated Stage 2 splits (train,calibration,development,confirmation).",
    )
    parser.add_argument(
        "--confirmation",
        action="store_true",
        help="Export the locked Stage 2 confirmation rollouts instead of development rollouts.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Analysis device such as cuda:0 or cpu (Stage 2 permits at most one GPU).",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command.startswith("stage2-") and not args.output_root:
        from .stage2_config import OUTPUT_RELATIVE

        args.output_root = os.path.join(_project_root(), OUTPUT_RELATIVE)
    if args.command.startswith("stage3-") and not args.output_root:
        from .stage3_config import OUTPUT_RELATIVE

        args.output_root = os.path.join(_project_root(), OUTPUT_RELATIVE)
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
        from .stage1_5 import (
            collect_old_test,
            finalize_stage1_5,
            prepare_old_test,
            screen_old_test,
        )

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
        elif args.command == "stage1-5-finalize":
            result = finalize_stage1_5(stage1_root, stage1_5_root)
        else:
            raise AssertionError(args.command)
    elif args.command.startswith("stage2-"):
        from .stage2 import (
            collect_candidates_task,
            collect_support_task,
            export_rollouts_zarr,
            finalize_freeze,
            freeze_task,
        )

        splits = tuple(
            value.strip()
            for value in (args.splits or "").split(",")
            if value.strip()
        )
        if args.command == "stage2-freeze-task":
            if not args.task_id:
                raise ValueError("--task-id is required")
            result = freeze_task(paths, output_root, args.task_id)
        elif args.command == "stage2-finalize-freeze":
            result = finalize_freeze(_project_root(), paths, output_root)
        elif args.command == "stage2-collect-support":
            if not args.task_id or not splits:
                raise ValueError("--task-id and --splits are required")
            result = {
                "shards": len(collect_support_task(paths, output_root, args.task_id, splits))
            }
        elif args.command == "stage2-collect-candidates":
            if not args.task_id or not splits:
                raise ValueError("--task-id and --splits are required")
            result = {
                "shards": len(collect_candidates_task(paths, output_root, args.task_id, splits))
            }
        elif args.command == "stage2-export-rollouts":
            result = export_rollouts_zarr(output_root, confirmation=args.confirmation)
        elif args.command == "stage2-run-development":
            from .stage2_analysis import run_development

            result = run_development(output_root, device_name=args.device)
        elif args.command == "stage2-refresh-summaries":
            from .stage2_analysis import refresh_derived_summaries

            result = refresh_derived_summaries(output_root, device_name=args.device)
        elif args.command == "stage2-verify":
            from .stage2_reporting import verify_stage2

            result = verify_stage2(_project_root(), output_root)
        else:
            raise AssertionError(args.command)
    elif args.command.startswith("stage3-"):
        splits = tuple(
            value.strip()
            for value in (args.splits or "").split(",")
            if value.strip()
        )
        if args.command == "stage3-freeze":
            from .stage3 import freeze_protocol

            result = freeze_protocol(_project_root(), paths, output_root)
        elif args.command == "stage3-refresh-freeze":
            from .stage3 import refresh_frozen_metadata

            result = refresh_frozen_metadata(output_root)
        elif args.command == "stage3-collect":
            from .stage3_collection import collect_task

            if not args.task_id or not splits:
                raise ValueError("--task-id and --splits are required")
            result = collect_task(
                _project_root(), paths, output_root, args.task_id, splits
            )
        elif args.command == "stage3-verify-training-reuse":
            from .stage3_collection import verify_training_reuse

            result = verify_training_reuse(_project_root(), output_root)
        else:
            raise AssertionError(args.command)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
