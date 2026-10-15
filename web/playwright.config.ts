import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 45_000,
  use: {
    baseURL: "http://127.0.0.1:8766",
    headless: true,
    channel: "chrome",
  },
  webServer: {
    command: "uv run --extra platform --extra train audioenv app --data-dir /tmp/audioenv-playwright-product --port 8766",
    cwd: "..",
    url: "http://127.0.0.1:8766/api/health",
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
