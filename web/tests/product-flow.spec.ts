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
  await page.getByRole("button", { name: "Reload verify" }).click();
  await expect(page.getByText(/Reload verified at step 20/)).toBeVisible();
  await page.getByRole("button", { name: "Export reproducibility bundle" }).click();
  await expect(page.getByRole("link", { name: "Download experiment export" })).toBeVisible();

  await page.goto("/artifacts");
  await expect(page.getByRole("link", { name: "Download" }).first()).toBeVisible();
  await page.goto("/evaluations");
  await expect(page.getByText(/24 samples/).first()).toBeVisible();
});
