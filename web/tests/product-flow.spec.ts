import { expect, test } from "@playwright/test";

test("create project, train, reload, compare, and export", async ({ page }) => {
  const suffix = Date.now().toString();
  await page.goto("/projects");
  await page.getByPlaceholder("Speech agent lab").fill(`Browser acceptance ${suffix}`);
  await page.getByPlaceholder("Optional context").fill("Playwright product flow");
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByText(`Browser acceptance ${suffix}`)).toBeVisible();

  await page.goto("/runs");
  await page.getByRole("button", { name: "New training run" }).click();
  await page.getByLabel("Project").selectOption({ label: `Browser acceptance ${suffix}` });
  await page.getByLabel("Run name").fill(`Browser run ${suffix}`);
  await page.getByLabel("Steps").fill("20");
  await page.getByRole("button", { name: "Run preflight" }).click();
  await expect(page.getByText("Preflight passed")).toBeVisible();
  await page.getByRole("checkbox").check();
  await page.getByRole("button", { name: "Launch experiment" }).click();
  await expect(page.getByText(`Browser run ${suffix}`)).toBeVisible();
  await page.getByText(`Browser run ${suffix}`).click();
  await expect(page.getByText("Status: completed"), "worker should finish").toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("Paired held-out comparison")).toBeVisible();
  await expect(page.getByText("Latest persisted rollout")).toBeVisible();
  await expect(page.getByText(/no reward event|milestone|terminal/).first()).toBeVisible();
  await page.getByRole("button", { name: "Reload verify" }).click();
  await expect(page.getByText(/Reload verified at step 20/)).toBeVisible();
  await page.getByRole("button", { name: "Resume 10 more steps" }).click();
  await expect(page.getByText(/Status: completed/), "resumed worker should finish").toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("30", { exact: true }).first()).toBeVisible();
  await page.getByRole("button", { name: "Export reproducibility bundle" }).click();
  await expect(page.getByRole("link", { name: "Download experiment export" })).toBeVisible();

  await page.goto("/artifacts");
  await expect(page.getByRole("link", { name: "Download" }).first()).toBeVisible();
  await page.goto("/evaluations");
  await expect(page.getByText(/24 samples/).first()).toBeVisible();
});

test("analyze repository and onboard an Omni model through the foundry", async ({ page }) => {
  await page.goto("/foundry");
  await page.getByLabel("Environment source").fill(".");
  await page.getByRole("button", { name: "Build environment harness" }).click();
  const environmentBuild = page.locator("details").filter({ hasText: "environment · coding_repository" }).first();
  await expect(environmentBuild).toBeVisible();
  await environmentBuild.locator("summary").click();
  await expect(environmentBuild.getByText(/configure and validate an OCI sandbox/)).toBeVisible();

  const modelId = `Qwen/Qwen3-Omni-browser-${Date.now()}`;
  await page.getByLabel("Foundry model ID").fill(modelId);
  await page.getByRole("button", { name: "Build model harness" }).click();
  const modelBuild = page.locator("details").filter({ hasText: modelId });
  await expect(modelBuild).toBeVisible();
  await modelBuild.locator("summary").click();
  await expect(modelBuild.getByText(/native-audio generation/)).toBeVisible();
});

test("configure the guarded RunPod quote workflow without provisioning", async ({ page, request }) => {
  const modelId = `playwright-qwen-${Date.now()}`;
  const registered = await request.post("/api/models", { data: {
    id: modelId, display_name: "Playwright Qwen3 Omni", adapter_type: "qwen3-omni",
    role: "both", config: { checkpoint: "/runpod-volume/models/qwen" },
  }});
  expect(registered.ok()).toBeTruthy();

  await page.goto("/runs");
  await page.getByRole("button", { name: "New training run" }).click();
  await page.getByLabel("Provider").selectOption("runpod-secure-cloud");
  await page.getByLabel("Agent").selectOption(modelId);
  await page.getByLabel("Caller").selectOption("audioenv/omni-caller");
  await page.getByLabel("Trainer").selectOption("qwen3-omni-thinker-gspo-lora");
  await page.getByLabel("Run name").fill("Playwright guarded GPU run");
  await page.getByLabel("Image digest").fill(`sha256:${"a".repeat(64)}`);
  await page.getByRole("button", { name: "Run preflight" }).click();
  await expect(page.getByText("Preflight passed")).toBeVisible();
  const experiments = await request.get("/api/experiments");
  const gpuRun = (await experiments.json()).find((item:any) => item.name === "Playwright guarded GPU run");
  expect(gpuRun.config.hyperparameters.max_steps).toBe(40);
  expect(gpuRun.config.hyperparameters.steps).toBeUndefined();
  await expect(page.getByRole("button", { name: "Request RunPod compute estimate" })).toBeEnabled();
  await expect(page.getByText(/provider preflight required/i)).toBeVisible();
});

test("configure and persist a guarded Modal estimate without provisioning", async ({ page, request }) => {
  const modelId = `playwright-modal-qwen-${Date.now()}`;
  const registered = await request.post("/api/models", { data: {
    id: modelId, display_name: "Playwright Modal Qwen", adapter_type: "qwen3-omni",
    role: "both", config: { checkpoint: "/workspace/audioenv/models/qwen" },
  }});
  expect(registered.ok()).toBeTruthy();

  await page.goto("/runs");
  await page.getByRole("button", { name: "New training run" }).click();
  await page.getByLabel("Provider").selectOption("modal-sandbox");
  await page.getByLabel("Agent").selectOption(modelId);
  await page.getByLabel("Caller").selectOption("audioenv/omni-caller");
  await page.getByLabel("Trainer").selectOption("qwen3-omni-thinker-gspo-lora");
  await page.getByLabel("Run name").fill("Playwright guarded Modal run");
  await page.getByLabel("Image digest").fill(`sha256:${"b".repeat(64)}`);
  await page.getByRole("button", { name: "Run preflight" }).click();
  await expect(page.getByText("Preflight passed")).toBeVisible();
  await page.getByRole("button", { name: "Request Modal compute estimate" }).click();
  await expect(page.getByText("Compute estimate ready for review")).toBeVisible();
  await expect(page.getByText(/operator-configured rate/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Provision Modal and launch" })).toBeDisabled();
});

test("configure local CUDA through the machine-preflight boundary", async ({ page, request }) => {
  const modelId = `playwright-local-cuda-qwen-${Date.now()}`;
  const registered = await request.post("/api/models", { data: {
    id: modelId, display_name: "Playwright Local CUDA Qwen", adapter_type: "qwen3-omni",
    role: "both", config: { checkpoint: "/srv/audioenv/models/qwen" },
  }});
  expect(registered.ok()).toBeTruthy();

  await page.goto("/runs");
  await page.getByRole("button", { name: "New training run" }).click();
  await page.getByLabel("Provider").selectOption("local-cuda");
  await page.getByLabel("Agent").selectOption(modelId);
  await page.getByLabel("Caller").selectOption("audioenv/omni-caller");
  await page.getByLabel("Trainer").selectOption("qwen3-omni-thinker-gspo-lora");
  await page.getByLabel("Run name").fill("Playwright local CUDA run");
  await page.getByLabel("GPU workspace root").fill("/srv/audioenv");
  await page.getByLabel("Image digest").fill(`sha256:${"c".repeat(64)}`);
  await page.getByRole("button", { name: "Run preflight" }).click();
  await expect(page.getByText("Preflight passed")).toBeVisible();
  await expect(page.getByRole("button", { name: "Inspect local CUDA machine" })).toBeEnabled();
  await expect(page.getByText(/provider preflight required/i)).toBeVisible();
});
