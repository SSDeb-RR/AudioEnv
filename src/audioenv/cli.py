"""Local authoring and validation CLI."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from importlib import resources
from pathlib import Path
import subprocess

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
    init_db = subparsers.add_parser("init-db", help="Initialize the local platform database")
    init_db.add_argument("--database", type=Path, default=Path(".audioenv/audioenv.db"))
    serve = subparsers.add_parser("serve", help="Run the local platform API")
    serve.add_argument("--database", type=Path, default=Path(".audioenv/audioenv.db"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    app_parser = subparsers.add_parser("app", help="Launch the production API and bundled dashboard")
    app_parser.add_argument("--data-dir", type=Path, default=Path(".audioenv"))
    app_parser.add_argument("--host", default="127.0.0.1")
    app_parser.add_argument("--port", type=int, default=8000)
    dev = subparsers.add_parser("dev", help="Run the local API and dashboard; workers launch per job")
    dev.add_argument("--database", type=Path, default=Path(".audioenv/audioenv.db"))
    dev.add_argument("--web-dir", type=Path, default=Path("web"))
    dev.add_argument("--api-port", type=int, default=8000)
    dev.add_argument("--web-port", type=int, default=5173)
    train_demo = subparsers.add_parser("train-demo", help="Run a real local CPU policy update")
    train_demo.add_argument("--output", type=Path, default=Path("outputs/tiny-training"))
    train_demo.add_argument("--steps", type=int, default=40)
    qwen_probe = subparsers.add_parser("qwen-probe", help="Probe Qwen3-Omni without downloading it")
    qwen_probe.add_argument("--checkpoint", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    platform_demo = subparsers.add_parser("platform-demo", help="Run the persisted local end-to-end workflow")
    platform_demo.add_argument("--root", type=Path, default=Path(".audioenv/demo"))
    platform_demo.add_argument("--steps", type=int, default=40)
    export_run = subparsers.add_parser("export-run", help="Export a persisted reproducibility record")
    export_run.add_argument("experiment_id")
    export_run.add_argument("--database", type=Path, default=Path(".audioenv/audioenv.db"))
    export_run.add_argument("--output", type=Path, required=True)
    harnesses = subparsers.add_parser("harnesses", help="List discovered harness plugins")
    harnesses.add_argument("--kind", choices=("environment", "model", "caller", "trainer", "provider"))
    foundry_environment = subparsers.add_parser("foundry-environment", help="Analyze a benchmark or repository")
    foundry_environment.add_argument("source_uri")
    foundry_environment.add_argument("--data-dir", type=Path, default=Path(".audioenv"))
    foundry_environment.add_argument(
        "--authoring-provider", choices=("deterministic", "codex"), default="deterministic"
    )
    foundry_model = subparsers.add_parser("foundry-model", help="Inspect an Omni model and select a family driver")
    foundry_model.add_argument("model_id")
    foundry_model.add_argument("--source")
    foundry_model.add_argument("--data-dir", type=Path, default=Path(".audioenv"))
    tau2_dataset = subparsers.add_parser("tau2-dataset", help="Build a hash-bound VeRL tau2 JSONL split")
    tau2_dataset.add_argument("--checkout", type=Path, required=True)
    tau2_dataset.add_argument("--output", type=Path, required=True)
    tau2_dataset.add_argument("--split", choices=("train", "test"), required=True)
    tau2_dataset.add_argument("--experiment-id", required=True)
    tau2_dataset.add_argument("--environment-hash", required=True)
    tau2_dataset.add_argument("--verifier-hash", required=True)
    tau2_dataset.add_argument("--policy-version", required=True)
    tau2_dataset.add_argument("--checkpoint-id", default="base")
    tau2_dataset.add_argument("--seed", type=int, action="append", dest="seeds")
    tau2_dataset.add_argument("--limit", type=int)
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
    if args.command == "init-db":
        from .platform.db import Database

        database = Database(f"sqlite:///{args.database}")
        database.create_schema()
        print(args.database.resolve())
        return 0
    if args.command == "serve":
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("Install AudioEnv with the 'platform' extra") from exc
        from .platform.api import create_app

        app = create_app(f"sqlite:///{args.database}")
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    if args.command == "app":
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("Install AudioEnv with the 'platform' extra") from exc
        from alembic import command
        from alembic.config import Config
        from .platform.api import create_app
        from .platform.db import Database, PlatformRepository

        data_dir = args.data_dir.resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        database_path = data_dir / "audioenv.db"
        database_url = f"sqlite:///{database_path}"
        source_migrations = Path(__file__).resolve().parents[2] / "migrations"
        packaged_migrations = Path(str(resources.files("audioenv").joinpath("migrations")))
        migrations = source_migrations if source_migrations.is_dir() else packaged_migrations
        alembic_config = Config()
        alembic_config.set_main_option("script_location", str(migrations))
        alembic_config.set_main_option("sqlalchemy.url", database_url)
        command.upgrade(alembic_config, "head")
        repository = PlatformRepository(Database(database_url))
        repository.database.create_schema()
        recovered = repository.recover_abandoned_jobs(stale_after_seconds=60)
        source_dist = Path(__file__).resolve().parents[2] / "web" / "dist"
        packaged_dist = Path(str(resources.files("audioenv").joinpath("web_dist")))
        static_dir = source_dist if source_dist.is_dir() else packaged_dist
        if not (static_dir / "index.html").is_file():
            raise RuntimeError("bundled dashboard is missing; run 'npm run build' in web/")
        print(f"AudioEnv: http://{args.host}:{args.port} (recovered jobs: {recovered})")
        uvicorn.run(create_app(database_url, static_dir=static_dir), host=args.host, port=args.port)
        return 0
    if args.command == "dev":
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("Install AudioEnv with the 'platform' extra") from exc
        from .platform.api import create_app

        web_dir = args.web_dir.resolve()
        if not (web_dir / "package.json").is_file():
            raise RuntimeError(f"dashboard package.json not found in {web_dir}")
        args.database.parent.mkdir(parents=True, exist_ok=True)
        dashboard = subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", str(args.web_port)],
            cwd=web_dir,
        )
        try:
            app = create_app(f"sqlite:///{args.database.resolve()}")
            uvicorn.run(app, host="127.0.0.1", port=args.api_port)
        finally:
            dashboard.terminate()
            try:
                dashboard.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dashboard.kill()
                dashboard.wait()
        return 0
    if args.command == "train-demo":
        from .training import TinyAudioPolicyTrainer, TinyTrainerConfig

        trainer = TinyAudioPolicyTrainer(package)
        config = TinyTrainerConfig(steps=args.steps)
        preflight = trainer.preflight(config)
        if not preflight.available:
            print(json.dumps(preflight.model_dump(mode="json"), indent=2))
            return 1
        result = trainer.train(config, args.output)
        payload = asdict(result)
        payload["checkpoint_path"] = str(result.checkpoint_path)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if result.initial_parameter_hash != result.final_parameter_hash else 1
    if args.command == "qwen-probe":
        from .models import Qwen3OmniAgentAdapter

        probe = Qwen3OmniAgentAdapter(args.checkpoint).probe()
        print(json.dumps(probe.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0 if probe.available else 1
    if args.command == "harnesses":
        from .harnesses import build_builtin_registry

        registry = build_builtin_registry()
        registry.discover_entry_points()
        print(json.dumps(
            [item.model_dump(mode="json") for item in registry.list(args.kind)],
            indent=2,
            sort_keys=True,
        ))
        return 0
    if args.command == "foundry-environment":
        from .artifacts import LocalArtifactStore
        from .foundry import EnvironmentFoundry

        data_dir = args.data_dir.resolve()
        result = EnvironmentFoundry(
            data_dir / "foundry" / "sources", LocalArtifactStore(data_dir / "artifacts")
        ).build(args.source_uri, authoring_provider=args.authoring_provider)
        print(json.dumps(result.report.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0 if result.report.status == "available" else 2
    if args.command == "foundry-model":
        from .artifacts import LocalArtifactStore
        from .foundry import ModelFoundry
        from .harnesses import build_builtin_registry

        data_dir = args.data_dir.resolve()
        report = ModelFoundry(
            build_builtin_registry(), LocalArtifactStore(data_dir / "artifacts")
        ).inspect(args.model_id, args.source or args.model_id)
        print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
        return 0 if report.status == "available" else 2
    if args.command == "tau2-dataset":
        from .adapters.tau2 import Tau2RetailSourceAdapter
        from .training.tau2_dataset import build_tau2_rollout_dataset

        with Tau2RetailSourceAdapter().open_worker(args.checkout.resolve(), timeout_seconds=120) as client:
            manifest = build_tau2_rollout_dataset(
                client, args.output.resolve(), split=args.split,
                experiment_id=args.experiment_id,
                environment_hash=args.environment_hash, verifier_hash=args.verifier_hash,
                policy_version=args.policy_version, checkpoint_id=args.checkpoint_id,
                seeds=tuple(args.seeds or (0,)), limit=args.limit,
            )
        print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
        return 0
    if args.command == "platform-demo":
        from .platform.db import Database, PlatformRepository, serialize_experiment
        from .platform.execution import LocalExecutionProvider

        args.root.mkdir(parents=True, exist_ok=True)
        database_path = args.root / "audioenv.db"
        database = Database(f"sqlite:///{database_path}")
        database.create_schema()
        repository = PlatformRepository(database)
        experiment = repository.create_experiment(
            "Local CPU training proof",
            {
                "environment_id": "audioenv/reference-agentic",
                "trainer_id": "audioenv/tiny-reinforce",
                "trainer_config": {"steps": args.steps},
            },
        )
        repository.transition(experiment.id, "validating")
        repository.transition(experiment.id, "queued")
        job = repository.create_job(
            experiment.id,
            "tiny_train",
            {"config": {"steps": args.steps}, "output": str(args.root / "runs" / experiment.id)},
        )
        submitted = LocalExecutionProvider().submit(database=database_path, job_id=job.id)
        stdout, stderr = submitted.process.communicate()
        final = repository.get_experiment(experiment.id)
        assert final is not None
        print(json.dumps({
            "experiment": serialize_experiment(final),
            "metrics": [
                {"step": row.step, "name": row.name, "value": row.value, "source": row.source}
                for row in repository.list_metrics(experiment.id)
            ],
            "worker_stdout": stdout,
            "worker_stderr": stderr,
        }, indent=2, sort_keys=True))
        return 0 if final.status == "completed" else 1
    if args.command == "export-run":
        from .platform.db import Database, PlatformRepository
        from .platform.export import export_experiment

        repository = PlatformRepository(Database(f"sqlite:///{args.database}"))
        print(export_experiment(repository, args.experiment_id, args.output))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

