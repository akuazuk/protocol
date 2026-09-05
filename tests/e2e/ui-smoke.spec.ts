import { test, expect, type Page } from '@playwright/test';

/**
 * Дымовые E2E-проверки интерфейса врача.
 *
 * Проверяется то, что не видно ни серверным тестам, ни логам: страница
 * отдаётся с кодом 200, но при этом часть скриптов заблокирована CSP или
 * не прошла проверку целостности - у врача просто не работает раздел
 * интерфейса, а сервер считает, что всё в порядке.
 *
 * Сервер поднимается с прод-политикой CSP, см. playwright.config.ts.
 */

/** Сообщения браузера, которые не считаем ошибкой интерфейса. */
function isIgnorableConsoleError(text: string): boolean {
  return (
    // Данных нет: сервер поднят в manifest-режиме без корпуса и без ключа API,
    // поэтому часть запросов аналитики законно отвечает ошибкой.
    /Failed to load resource.*\b(404|500|503)\b/i.test(text) ||
    /favicon/i.test(text)
  );
}

interface PageProblems {
  cspViolations: string[];
  consoleErrors: string[];
  failedRequests: string[];
}

/** Подписка на нарушения CSP, ошибки консоли и упавшие запросы. */
function collectProblems(page: Page): PageProblems {
  const problems: PageProblems = { cspViolations: [], consoleErrors: [], failedRequests: [] };

  page.on('console', (msg) => {
    if (msg.type() !== 'error') return;
    const text = msg.text();
    // Chromium сообщает о блокировке CSP именно через консоль.
    if (/Content Security Policy|Refused to (load|execute|apply|connect)/i.test(text)) {
      problems.cspViolations.push(text);
      return;
    }
    // Провал SRI выглядит как отказ выполнить скрипт из-за несовпадения хеша.
    if (/integrity|Failed to find a valid digest/i.test(text)) {
      problems.cspViolations.push(text);
      return;
    }
    if (!isIgnorableConsoleError(text)) {
      problems.consoleErrors.push(text);
    }
  });

  page.on('requestfailed', (req) => {
    const failure = req.failure()?.errorText || '';
    problems.failedRequests.push(`${req.url()} - ${failure}`);
  });

  return problems;
}

test.describe('Интерфейс врача', () => {
  test('главная страница загружается с прод-CSP без блокировок', async ({ page }) => {
    const problems = collectProblems(page);

    const response = await page.goto('/', { waitUntil: 'domcontentloaded' });
    expect(response?.status(), 'страница должна отдаваться со 200').toBe(200);

    await expect(page).toHaveTitle(/Навигатор клинических протоколов/);

    // Сначала CSP: если политика режет свои же скрипты, остальные проверки
    // будут падать по следствию, а не по причине.
    expect(
      problems.cspViolations,
      'CSP или SRI заблокировали ресурс интерфейса - страница отдана, но UI неполный'
    ).toEqual([]);
  });

  test('прод-CSP действительно отдаётся в заголовках', async ({ page }) => {
    const response = await page.goto('/', { waitUntil: 'domcontentloaded' });
    const headers = response?.headers() || {};
    const csp = headers['content-security-policy'] || headers['content-security-policy-report-only'];

    expect(csp, 'нет заголовка CSP: ENABLE_DEFAULT_CSP не применился').toBeTruthy();
    // Ровно те директивы, на которые рассчитывает интерфейс.
    expect(csp).toContain("object-src 'none'");
    expect(csp).toContain("base-uri 'self'");
    expect(csp).toContain('https://cdn.jsdelivr.net');
  });

  test('chart.js загружается с проверкой целостности (SRI)', async ({ page }) => {
    // Прямая проверка SRI: при неверном integrity браузер откажется выполнять
    // скрипт, глобального Chart не будет, и графики методслужбы не отрисуются.
    // Тихая поломка - поэтому проверка явная.
    await page.goto('/', { waitUntil: 'load' });

    const chartLoaded = await page.evaluate(
      () => typeof (window as unknown as { Chart?: unknown }).Chart !== 'undefined'
    );
    expect(
      chartLoaded,
      'window.Chart отсутствует: CDN недоступен либо не сошёлся хеш integrity'
    ).toBe(true);
  });

  test('собственные стили и скрипты страницы отдаются', async ({ page }) => {
    const problems = collectProblems(page);
    await page.goto('/', { waitUntil: 'load' });

    // Ассеты из корневой раздачи закрыты allow-list (см. rag_server.py):
    // опечатка в списке = 404 на свой же файл.
    const ownAssetFailures = problems.failedRequests.filter((u) => !u.includes('://cdn.'));
    expect(ownAssetFailures, 'не отдался собственный ресурс интерфейса').toEqual([]);
  });

  test('ключевые элементы интерфейса присутствуют', async ({ page }) => {
    await page.goto('/', { waitUntil: 'domcontentloaded' });
    await expect(page.locator('#app-chrome')).toBeAttached();
    await expect(page.locator('#app-main-tabs')).toBeAttached();
  });

  test('health и version отвечают', async ({ request }) => {
    const live = await request.get('/health/live');
    expect(live.ok()).toBeTruthy();

    const version = await request.get('/api/version');
    expect(version.ok()).toBeTruthy();
    const body = await version.json();
    // Версия обязана быть непустой: по ней release-координатор подтверждает деплой.
    expect(String(body.version || '')).not.toBe('');
  });
});
