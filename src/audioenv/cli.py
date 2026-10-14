"""Local authoring and validation CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .export import export_package, export_trace
from .reference import build_reference_package
from .runtime import AudioEnvironment
from .spec import SpeakAction, ToolCallAction
from .validation import validate_package


def _task_id(domain: str) -> str:
    return {"retail": "retail-refund-001", "telecom": "telecom-roaming-001"}[domain]


def _run_demo(domain: str, trajectory: str, output: Path | None) -> int:
    package = build_reference_package()
    task = package.task(_task_id(domain))
    env = AudioEnvironment(package)
    env.reset(task.id, seed=7, model_id=trajectory)
    if trajectory == "oracle":
        for action in task.oracle_actions:
            env.step(action)
    elif trajectory == "wrong":
        expected = task.required_tools[0]
        arguments = dict(expected.arguments)
        arguments[next(iter(arguments))] = "MISHEARD-VALUE"
        env.step(ToolCallAction(expected.name, arguments))
        env.step(SpeakAction(b"agent-audio", "I have completed that.", True))
    else:
        env.step(SpeakAction(b"agent-audio", "I cannot help.", True))
    if output:
        export_trace(env.trace, output, include_observability=False)
    print(env.trace.to_json(include_observability=False))
    return 0 if bool(env.trace.final_metrics.get("terminal_pass")) == (trajectory == "oracle") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="audioenv")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="Validate the bundled reference environment")
    export_parser = subparsers.add_parser("export", help="Export a leakage-safe package")
    export_parser.add_argument("destination", type=Path)
    demo = subparsers.add_parser("demo", help="Run a reference trajectory")
    demo.add_argument("--domain", choices=("retail", "telecom"), default="retail")
    demo.add_argument("--trajectory", choices=("oracle", "noop", "wrong"), default="oracle")
    demo.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    package = build_reference_package()
    if args.command == "validate":
        report = validate_package(package)
        for check in report.checks:
            print(f"{'PASS' if check.passed else 'FAIL'} {check.name}: {check.details}")
        return 0 if report.passed else 1
    if args.command == "export":
        print(export_package(package, args.destination))
        return 0
    if args.command == "demo":
        return _run_demo(args.domain, args.trajectory, args.output)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

