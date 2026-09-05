import { defineConfig, devices } from '@playwright/test';

/**
 * E2E-конфигурация Protocol.
 *
 * Сервер поднимается с ENABLE_DEFAULT_CSP=1 - то есть ровно с той политикой
 * безопасности, что включена в проде (deploy/gcp-app/deploy_to_gce.sh).
 * Смысл именно в этом: CSP, которая блокирует собственные скрипты интерфейса,
 * ломает страницу молча - в логах сервера ничего нет, 200 отдаётся, а у врача
 * не работает часть UI. Такое ловится только браузером.
 *
 * RAG_STARTUP_MODE=manifest - быстрый старт без загрузки полного корпуса:
 * проверяется отдача и целостность страницы, а не качество поиска
 * (оно в eval/, см. scripts/ops/run_search_quality_gate.sh).
 */
const PORT = Number(process.env.E2E_PORT || 8099);
const BASE_URL = `http://127.0.0.1:${PORT}`;

export default defineConfig({
  testDir: './tests/e2e',
  // Клиника: тест, который «иногда проходит», хуже отсутствующего.
  // Повторы только в CI и только чтобы отсечь сетевые срывы к CDN.
  retries: process.env.CI ? 1 : 0,
  forbidOnly: !!process.env.CI,
  reporter: process.env.CI ? [['github'], ['list']] : [['list']],
  timeout: 60_000,
  expect: { timeout: 10_000 },

  use: {
    baseURL: BASE_URL,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },

  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],

  webServer: {
    command: `python3 -m uvicorn rag_server:app --host 127.0.0.1 --port ${PORT}`,
    url: `${BASE_URL}/health/live`,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
    stdout: 'pipe',
    stderr: 'pipe',
    env: {
      // Прод-политика безопасности: главное, что проверяет этот прогон.
      ENABLE_DEFAULT_CSP: '1',
      // Без ключа API и без обращений к внешней модели.
      RAG_GEMINI_EMBED_RERANK: '0',
      RAG_STARTUP_MODE: 'manifest',
      RAG_LAZY_CHUNK_STORE: '1',
      PYTHONUNBUFFERED: '1',
    },
  },
});
