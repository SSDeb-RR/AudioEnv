import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as Tooltip from "@radix-ui/react-tooltip";
import {
  Activity, Archive, Boxes, BrainCircuit, CircleGauge, ClipboardCheck,
  Database, FlaskConical, Settings, TriangleAlert,
} from "lucide-react";
import { Link, NavLink, Route, Routes, useParams } from "react-router-dom";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip as ChartTooltip, XAxis, YAxis } from "recharts";
import { useState } from "react";

type Experiment = {
  id: string; name: string; status: string; created_at: string; failure_reason: string | null;
};
type Overview = {
  total_experiments: number; active_runs: number; failed_runs: number;
  recent: Experiment[]; compute_cost_usd: number | null; compute_cost_status: string;
};
type Metric = { id: number; step: number; name: string; value: number; source: string };
type Evaluation = { id: string; experiment_id: string; checkpoint_id: string; split: string; metrics: Record<string, number | string> };
type ModelRecord = { id: string; display_name: string; adapter_type: string; role: string; available: boolean; probe: { reasons?: string[] } | null };
type EnvironmentRecord = { id: string; version: string; content_hash: string; source_kind: string; authoring_status: string };

const nav = [
  ["/", "Overview", CircleGauge], ["/models", "Models", BrainCircuit],
  ["/benchmarks", "Benchmarks & Tasks", Database], ["/environments", "Environments", Boxes],
  ["/runs", "Training Runs", Activity], ["/evaluations", "Evaluations", ClipboardCheck],
  ["/artifacts", "Artifacts", Archive], ["/settings", "Settings", Settings],
] as const;

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}

function Shell({ children }: { children: React.ReactNode }) {
  return <div className="shell">
    <aside>
      <div className="brand"><span className="mark"><FlaskConical size={18} /></span><div>AudioEnv<small>Training foundry</small></div></div>
      <nav>{nav.map(([to, label, Icon]) => <NavLink key={to} to={to} end={to === "/"}><Icon size={17} />{label}</NavLink>)}</nav>
      <div className="runtime"><span className="dot" />Local runtime<small>Single-user preview</small></div>
    </aside>
    <main>{children}</main>
  </div>;
}

function Header({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) {
  return <header><div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1><p className="description">{description}</p></div><span className="preview">V1 preview</span></header>;
}

function OverviewPage() {
  const query = useQuery({ queryKey: ["overview"], queryFn: () => getJson<Overview>("/api/overview") });
  const evaluations = useQuery({ queryKey: ["evaluations"], queryFn: () => getJson<Evaluation[]>("/api/evaluations") });
  if (query.isError) return <ErrorState message={(query.error as Error).message} />;
  const data = query.data;
  return <><Header eyebrow="Workspace" title="Training overview" description="Actual experiment state from the local AudioEnv service." />
    <section className="stats">
      <Stat label="Experiments" value={data?.total_experiments ?? "—"} />
      <Stat label="Active runs" value={data?.active_runs ?? "—"} accent />
      <Stat label="Needs attention" value={data?.failed_runs ?? "—"} warning={(data?.failed_runs ?? 0) > 0} />
      <Tooltip.Provider><Tooltip.Root><Tooltip.Trigger asChild><div><Stat label="Compute cost" value={data?.compute_cost_usd == null ? "Unavailable" : `$${data.compute_cost_usd.toFixed(2)}`} /></div></Tooltip.Trigger><Tooltip.Portal><Tooltip.Content className="tip">No provider price has been configured.</Tooltip.Content></Tooltip.Portal></Tooltip.Root></Tooltip.Provider>
    </section>
    <section className="panel"><div className="panel-head"><div><h2>Recent activity</h2><p>Runs are never synthesized by the dashboard.</p></div><Link to="/runs">View all</Link></div>
      {!data?.recent.length ? <Empty title="No experiments yet" body="Create an experiment through the API or run the CPU training demo." /> : <RunTable runs={data.recent} />}
    </section>
    <section className="split"><div className="panel"><h2>Learning lift</h2>{evaluations.data?.length ? <Comparison evaluation={evaluations.data[0]} /> : <Empty title="No comparison available" body="Baseline and trained evaluations are required before a lift is shown." />}</div><div className="panel"><h2>System status</h2><StatusRow label="Local API" status="Available" /><StatusRow label="CUDA worker" status="Not configured" /><StatusRow label="Cost telemetry" status="Unavailable" /></div></section>
  </>;
}

function RunsPage() {
  const query = useQuery({ queryKey: ["runs"], queryFn: () => getJson<Experiment[]>("/api/experiments") });
  const [creating, setCreating] = useState(false);
  return <><Header eyebrow="Execution" title="Training runs" description="Validation, rollout, training, and evaluation lifecycle." /><div className="actions"><button onClick={() => setCreating(!creating)}>{creating ? "Close setup" : "New local run"}</button></div>{creating && <NewRun onDone={() => setCreating(false)} />}<section className="panel">{query.data?.length ? <RunTable runs={query.data} /> : <Empty title="No training runs" body="Launch requires a validated environment, model probe, trainer, and explicit review." />}</section></>;
}

function NewRun({ onDone }: { onDone: () => void }) {
  const client = useQueryClient();
  const [name, setName] = useState("Local CPU training proof");
  const [steps, setSteps] = useState(40);
  const mutation = useMutation({
    mutationFn: async () => {
      const experiment = await postJson<Experiment>("/api/experiments", { name, config: { environment_id: "audioenv/reference-agentic", trainer_id: "audioenv/tiny-reinforce", trainer_config: { steps } } });
      return postJson(`/api/experiments/${experiment.id}/launch`, { confirmed: true, provider: "local-subprocess" });
    },
    onSuccess: () => { client.invalidateQueries({ queryKey: ["runs"] }); client.invalidateQueries({ queryKey: ["overview"] }); onDone(); },
  });
  return <section className="panel setup"><div><span>1 · Environment</span><b>Reference agentic fixture</b><small>Validated local package</small></div><div><span>2 · Agent & caller</span><b>Tiny audio policy · scripted caller</b><small>Mechanics proof only</small></div><div><span>3 · Trainer</span><b>REINFORCE · Torch CPU</b><label>Steps<input type="number" min="1" max="1000" value={steps} onChange={event => setSteps(Number(event.target.value))} /></label></div><div><span>4 · Review & launch</span><input aria-label="Run name" value={name} onChange={event => setName(event.target.value)} /><small>Hardware: local CPU · Estimated cost: unavailable</small><button disabled={mutation.isPending || !name || steps < 1} onClick={() => mutation.mutate()}>{mutation.isPending ? "Launching…" : "Validate & launch"}</button>{mutation.isError && <small className="form-error">{(mutation.error as Error).message}</small>}</div></section>;
}

function RunPage() {
  const { id = "" } = useParams();
  const run = useQuery({ queryKey: ["run", id], queryFn: () => getJson<Experiment>(`/api/experiments/${id}`) });
  const metrics = useQuery({ queryKey: ["metrics", id], queryFn: () => getJson<Metric[]>(`/api/experiments/${id}/metrics`) });
  const points = metrics.data?.filter(item => item.name === "loss") ?? [];
  return <><Header eyebrow="Training run" title={run.data?.name ?? "Loading…"} description={`Status: ${run.data?.status ?? "unknown"}`} /><section className="panel chart"><h2>Reported loss</h2>{points.length ? <ResponsiveContainer width="100%" height={280}><LineChart data={points}><CartesianGrid stroke="#252b34" /><XAxis dataKey="step" stroke="#77808f" /><YAxis stroke="#77808f" /><ChartTooltip /><Line type="monotone" dataKey="value" stroke="#77d9bd" dot={false} /></LineChart></ResponsiveContainer> : <Empty title="Loss unavailable" body="This trainer has not reported loss metrics." />}</section></>;
}

function ResourcePage({ title, endpoint, description }: { title: string; endpoint: string; description: string }) {
  const query = useQuery({ queryKey: [endpoint], queryFn: () => getJson<unknown[]>(endpoint) });
  return <><Header eyebrow="Registry" title={title} description={description} /><section className="panel">{query.data?.length ? <pre>{JSON.stringify(query.data, null, 2)}</pre> : <Empty title={`No ${title.toLowerCase()}`} body="Nothing has been registered through the backend yet." />}</section></>;
}

function ModelsPage() {
  const client = useQueryClient();
  const query = useQuery({ queryKey: ["models"], queryFn: () => getJson<ModelRecord[]>("/api/models") });
  const [modelId, setModelId] = useState("Qwen/Qwen3-Omni-30B-A3B-Instruct");
  const [role, setRole] = useState("agent");
  const register = useMutation({
    mutationFn: () => postJson("/api/models", { id: modelId, display_name: modelId.split("/").pop(), adapter_type: "qwen3-omni", role, config: { checkpoint: modelId } }),
    onSuccess: () => client.invalidateQueries({ queryKey: ["models"] }),
  });
  const probe = useMutation({
    mutationFn: (id: string) => postJson("/api/models/probe", { model_id: id }),
    onSuccess: () => client.invalidateQueries({ queryKey: ["models"] }),
  });
  return <><Header eyebrow="Registry" title="Models" description="Register identifiers, then probe the actual adapter and hardware before use." /><section className="panel model-form"><label>Model or local checkpoint<input value={modelId} onChange={event => setModelId(event.target.value)} /></label><label>Role<select value={role} onChange={event => setRole(event.target.value)}><option value="agent">Trainable agent</option><option value="caller">Frozen caller</option><option value="both">Both</option></select></label><button disabled={!modelId || register.isPending} onClick={() => register.mutate()}>Register, not load</button></section><section className="panel">{query.data?.length ? <div className="model-list">{query.data.map(model => <div key={model.id}><span><b>{model.display_name}</b><small>{model.id} · {model.role}</small></span><i className={`status ${model.available ? "running" : ""}`}>{model.available ? "Available" : "Not verified"}</i><button onClick={() => probe.mutate(model.id)}>Probe</button>{model.probe?.reasons?.length ? <small>{model.probe.reasons.join(" · ")}</small> : null}</div>)}</div> : <Empty title="No models" body="Entering an identifier does not make a checkpoint available. Register and probe it here." />}</section></>;
}

function EnvironmentsPage() {
  const client = useQueryClient();
  const query = useQuery({ queryKey: ["environments"], queryFn: () => getJson<EnvironmentRecord[]>("/api/environments") });
  const [open, setOpen] = useState(false);
  const [environmentId, setEnvironmentId] = useState("custom/workflow");
  const [taskId, setTaskId] = useState("workflow-1");
  const [goal, setGoal] = useState("Complete the workflow");
  const create = useMutation({
    mutationFn: () => postJson("/api/environments/custom", {
      environment_id: environmentId, description: goal, task_id: taskId, goal,
      initial_state: { task: { status: "pending" } }, expected_state: { task: { status: "complete" } },
      initial_utterance: goal, tools: [{ name: "complete_task", description: "Complete the workflow", operation: "set_path", path: ["task", "status"], required_arguments: ["status"], value_argument: "status" }],
      required_tools: [{ name: "complete_task", arguments: { status: "complete" }, milestone: "workflow_complete" }],
      required_communications: ["complete"],
    }),
    onSuccess: () => { client.invalidateQueries({ queryKey: ["environments"] }); setOpen(false); },
  });
  return <><Header eyebrow="Construction" title="Environments" description="Import, author, validate, and version executable task worlds." /><section className="mode-grid"><div className="mode"><b>Existing benchmark</b><p>τ² retail source importer is available; executable worker integration remains gated.</p><span>Partial</span></div><div className="mode active"><b>Custom task</b><p>Create a constrained state-transition environment and run its verifier suite.</p><button onClick={() => setOpen(!open)}>{open ? "Close" : "Create"}</button></div><div className="mode"><b>Repository task</b><p>Disabled until an OCI sandbox passes the security capability check.</p><span>Unsupported</span></div></section>{open && <section className="panel environment-form"><label>Environment ID<input value={environmentId} onChange={event => setEnvironmentId(event.target.value)} /></label><label>Task ID<input value={taskId} onChange={event => setTaskId(event.target.value)} /></label><label>Goal<input value={goal} onChange={event => setGoal(event.target.value)} /></label><button disabled={create.isPending || !environmentId || !taskId || !goal} onClick={() => create.mutate()}>{create.isPending ? "Validating…" : "Create & validate"}</button>{create.isError && <small className="form-error">{(create.error as Error).message}</small>}</section>}<section className="panel">{query.data?.length ? <div className="model-list">{query.data.map(item => <div key={item.id}><span><b>{item.id}</b><small>{item.content_hash.slice(0, 14)} · {item.source_kind}</small></span><i className="status running">{item.authoring_status}</i></div>)}</div> : <Empty title="No environments" body="Create a custom package or import a supported benchmark source." />}</section></>;
}

function App() {
  return <Shell><Routes>
    <Route path="/" element={<OverviewPage />} />
    <Route path="/models" element={<ModelsPage />} />
    <Route path="/benchmarks" element={<ResourcePage title="Benchmarks & tasks" endpoint="/api/environments" description="Imported sources and immutable task splits." />} />
    <Route path="/environments" element={<EnvironmentsPage />} />
    <Route path="/runs" element={<RunsPage />} /><Route path="/runs/:id" element={<RunPage />} />
    <Route path="/evaluations" element={<ResourcePage title="Evaluations" endpoint="/api/evaluations" description="Traceable baseline and checkpoint comparisons." />} />
    <Route path="/artifacts" element={<ResourcePage title="Artifacts" endpoint="/api/artifacts" description="Checkpoints, traces, audio, logs, and reproducibility exports." />} />
    <Route path="/settings" element={<><Header eyebrow="Workspace" title="Settings" description="Local execution and storage configuration." /><section className="panel"><StatusRow label="Deployment" status="Local, single user" /><StatusRow label="Authentication" status="Not configured" /><StatusRow label="Artifact storage" status="Local filesystem" /></section></>} />
  </Routes></Shell>;
}

function Stat({ label, value, accent, warning }: { label: string; value: string | number; accent?: boolean; warning?: boolean }) { return <div className={`stat ${accent ? "accent" : ""} ${warning ? "warning" : ""}`}><span>{label}</span><strong>{value}</strong></div>; }
function Empty({ title, body }: { title: string; body: string }) { return <div className="empty"><FlaskConical size={22} /><strong>{title}</strong><p>{body}</p></div>; }
function ErrorState({ message }: { message: string }) { return <div className="empty error"><TriangleAlert /><strong>Backend unavailable</strong><p>{message}</p></div>; }
function StatusRow({ label, status }: { label: string; status: string }) { return <div className="status-row"><span>{label}</span><b>{status}</b></div>; }
function RunTable({ runs }: { runs: Experiment[] }) { return <div className="table"><div className="tr heading"><span>Run</span><span>Status</span><span>Created</span></div>{runs.map(run => <Link className="tr" key={run.id} to={`/runs/${run.id}`}><span><b>{run.name}</b><small>{run.id.slice(0, 8)}</small></span><span><i className={`status ${run.status}`}>{run.status}</i></span><span>{new Date(run.created_at).toLocaleString()}</span></Link>)}</div>; }
function Comparison({ evaluation }: { evaluation: Evaluation }) {
  const baseline = Number(evaluation.metrics.baseline_success_rate ?? 0);
  const trained = Number(evaluation.metrics.trained_success_rate ?? 0);
  return <div className="comparison"><div><span>Baseline</span><b>{Math.round(baseline * 100)}%</b><i><em style={{ width: `${baseline * 100}%` }} /></i></div><div><span>Trained</span><b>{Math.round(trained * 100)}%</b><i><em style={{ width: `${trained * 100}%` }} /></i></div><small>{evaluation.metrics.claim_scope === "cpu_mechanics_proof" ? "CPU mechanics proof · not a Qwen result" : evaluation.split}</small></div>;
}

export default App;
