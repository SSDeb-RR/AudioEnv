from dataclasses import replace
from pathlib import Path
import time

from fastapi.testclient import TestClient

from audioenv.contracts import EvaluationProtocolV1, EvaluationSampleV1
from audioenv.evaluation import build_evaluation_result, compare_evaluations
from audioenv.platform.api import create_app
from audioenv.platform.api import _gpu_artifact_kind, _public_trace_row
from audioenv.platform.runpod import RunPodConfig, RunPodSecureCloudProvider
from audioenv.reference import build_reference_package, build_reference_split_manifest
from audioenv.spec import SpeakAction
from audioenv.training.tiny import ActionCandidateBuilder


def test_reference_v2_split_is_frozen_and_disjoint():
    package = build_reference_package()
    manifest = build_reference_split_manifest(package)
    assert len(package.tasks) == 48
    assert (len(manifest.train), len(manifest.validation), len(manifest.test)) == (32, 8, 8)
    assert not (set(manifest.train) & set(manifest.validation))
    assert not (set(manifest.train) & set(manifest.test))
    assert not (set(manifest.validation) & set(manifest.test))


def test_candidate_builder_does_not_read_oracle_actions():
    package = build_reference_package()
    original = package.tasks[0]
    poisoned = replace(original, oracle_actions=(SpeakAction(b"poison", "verifier leak", True),))
    candidates = ActionCandidateBuilder(package).build(poisoned)
    assert candidates == package.public_action_candidates[original.id][:2]
    assert poisoned.oracle_actions[0] not in candidates


def test_comparison_rejects_protocol_drift():
    digest = "a" * 64
    protocol = EvaluationProtocolV1(
        environment_hash=digest, verifier_hash=digest, split_manifest_hash=digest,
        split="test", caller_hash=digest, audio_pipeline_hash=digest,
        sampling_hash=digest, checkpoint_id="base", seeds=(1,),
    )
    sample = EvaluationSampleV1(task_id="t", audio_profile_id="clean", seed=1, passed=False, reward=0)
    baseline = build_evaluation_result(protocol, (sample,))
    drifted = build_evaluation_result(
        protocol.model_copy(update={"checkpoint_id": "trained", "caller_hash": "b" * 64}),
        (sample.model_copy(update={"passed": True, "reward": 1}),),
    )
    comparison = compare_evaluations(baseline, drifted)
    assert not comparison.compatible
    assert comparison.incompatibilities == ("caller_hash",)


def test_native_v2_trace_normalizes_without_private_training_tensors():
    value = {
        "schema_version": "audioenv.trajectory/v2", "task_id": "tau2-1", "seed": 7,
        "processor_config": {"audio_profile_id": "native"},
        "terminal_reward": 1.0, "process_reward": 0.25, "done": True,
        "turns": [{"action": {"kind": "tool_call", "name": "lookup"},
                   "generated_ids": [11, 12], "rollout_logprobs": [-1.0, -2.0]}],
    }
    public = _public_trace_row(value)
    assert public["trace"]["final_metrics"]["terminal_pass"] is True
    assert public["trace"]["final_metrics"]["latency_available"] is False
    assert "generated_ids" not in str(public)
    assert "rollout_logprobs" not in str(public)
    assert _gpu_artifact_kind(Path("incoming/trajectories/trajectory-42.jsonl")) == "training_trajectories"


class FakeRunPod:
    def __init__(self): self.stopped = []
    def list_gpu_types(self): return [{"id":"H100-80","memory_gb":80,"secure_cloud_available":8,"price_per_gpu_hour":2.0}]
    def create_pod(self, payload): return {"id":"pod-1","payload":payload}
    def get_pod(self, pod_id): return {"id":pod_id,"status":"RUNNING"}
    def stop_pod(self, pod_id): self.stopped.append(pod_id); return {"id":pod_id,"status":"STOPPED"}


def test_runpod_requires_typed_confirmation_and_enforces_budget():
    transport = FakeRunPod()
    provider = RunPodSecureCloudProvider(transport)
    config = RunPodConfig(
        run_name="gpu-gate", image_digest="sha256:" + "a" * 64,
        max_runtime_hours=4, max_total_budget_usd=100,
    )
    quote = provider.quote(config)
    assert quote.available and quote.maximum_cost_usd == 64
    try:
        provider.provision(config, quote, confirmation_name="wrong")
        raise AssertionError("provision should reject incorrect confirmation")
    except ValueError:
        pass
    assert provider.provision(config, quote, confirmation_name="gpu-gate")["id"] == "pod-1"
    provider.enforce_limits("pod-1", config, quote, elapsed_hours=4)
    assert transport.stopped == ["pod-1"]


def test_api_launch_train_reload_compare_and_serve_spa(tmp_path):
    static = Path(__file__).parents[1] / "web" / "dist"
    app = create_app(f"sqlite:///{tmp_path / 'product.sqlite'}", static_dir=static)
    client = TestClient(app)
    assert "AudioEnv" in client.get("/").text
    project = client.post("/api/projects", json={"name":"Acceptance","description":"E2E"}).json()
    environment = client.get("/api/environments").json()[0]
    model = next(item for item in client.get("/api/models").json() if item["available"])
    trainer = next(item for item in client.get("/api/trainers").json() if item["status"] == "available")
    experiment = client.post("/api/experiments", json={
        "name":"Acceptance run", "project_id":project["id"],
        "config":{
            "project_id":project["id"], "environment_id":environment["id"],
            "agent_model_id":model["id"], "trainer_id":trainer["id"],
            "provider_id":"local-subprocess", "reward_version":"reference-v2",
            "hyperparameters":{"steps":20,"checkpoint_interval":10,"seed":7,"learning_rate":0.08},
        },
    }).json()
    preflight = client.post(f"/api/experiments/{experiment['id']}/preflight", json={}).json()
    assert preflight["available"]
    launched = client.post(
        f"/api/experiments/{experiment['id']}/launch",
        json={"confirmed":True,"provider":"local-subprocess"},
    )
    assert launched.status_code == 202
    for _ in range(100):
        current = client.get(f"/api/experiments/{experiment['id']}").json()
        if current["status"] in {"completed","failed","cancelled"}: break
        time.sleep(0.05)
    assert current["status"] == "completed", current
    artifacts = [a for a in client.get("/api/artifacts").json() if a["experiment_id"] == experiment["id"]]
    checkpoint = next(a for a in artifacts if a["kind"] == "checkpoint")
    verified = client.post(f"/api/experiments/{experiment['id']}/checkpoints/{checkpoint['id']}/verify", json={}).json()
    assert verified["verified"]
    resumed = client.post(f"/api/experiments/{experiment['id']}/resume", json={
        "checkpoint_artifact_id": checkpoint["id"],
        "run_name": "Acceptance run resumed", "additional_steps": 2,
    })
    assert resumed.status_code == 201
    resumed_run = resumed.json()
    assert resumed_run["config"]["resume_parent_experiment_id"] == experiment["id"]
    assert resumed_run["config"]["resume_checkpoint_artifact_id"] == checkpoint["id"]
    assert resumed_run["config"]["hyperparameters"]["steps"] == 2
    assert client.post(f"/api/experiments/{resumed_run['id']}/preflight", json={}).json()["available"]
    assert client.post(
        f"/api/experiments/{resumed_run['id']}/launch",
        json={"confirmed": True, "provider": "local-subprocess"},
    ).status_code == 202
    for _ in range(100):
        resumed_current = client.get(f"/api/experiments/{resumed_run['id']}").json()
        if resumed_current["status"] in {"completed", "failed", "cancelled"}: break
        time.sleep(0.05)
    assert resumed_current["status"] == "completed", resumed_current
    resumed_metrics = client.get(f"/api/experiments/{resumed_run['id']}/metrics").json()
    assert max(item["step"] for item in resumed_metrics) == 22
    evaluations = [e for e in client.get("/api/evaluations").json() if e["experiment_id"] == experiment["id"]]
    assert len(evaluations) == 2
    comparison = client.get(f"/api/comparisons?experiment_id={experiment['id']}").json()[0]["result"]
    assert comparison["compatible"] and comparison["paired_sample_count"] == 24
    traces = client.get(f"/api/experiments/{experiment['id']}/traces").json()
    assert traces["available"] and len(traces["traces"]) == 20
    assert traces["traces"][-1]["training_step"] == 20
    assert all(
        step["observability"] == {}
        for row in traces["traces"] for step in row["trace"]["steps"]
    )
    exported = client.post(f"/api/experiments/{experiment['id']}/export", json={})
    assert exported.status_code == 201
    assert client.get(exported.json()["download_url"]).status_code == 200

    trace_artifact = app.state.repository.get_artifact(traces["artifact_id"])
    Path(trace_artifact.path).write_bytes(b"tampered")
    assert client.get(f"/api/experiments/{experiment['id']}/traces").status_code == 409
