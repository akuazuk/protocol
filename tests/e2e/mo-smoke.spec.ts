import { test, expect, type Page } from '@playwright/test';

// Real MO HTML/assets/CSP; synthetic API only. No clinical records or model calls.
const pages = ['yesterday', 'overview', 'queue', 'documents', 'doctors', 'medications', 'labs', 'reports', 'kp-sync', 'rceth-sync', 'settings'];

async function mockMo(page: Page, failFamily = false) {
  const requests: URL[] = [];
  const problems: string[] = [];
  page.on('pageerror', error => problems.push(error.message));
  page.on('console', msg => {
    if (/Content Security Policy|Refused to (load|execute|apply)|Failed to find a valid digest/i.test(msg.text())) problems.push(msg.text());
  });
  await page.addInitScript(() => localStorage.setItem('protocol_methodist_token', 'synthetic-local-only'));
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url());
    requests.push(url);
    const p = url.pathname;
    let data: unknown = { ok: true, items: [], rows: [], facets: {}, data_through: '2026-08-02' };
    if (p.endsWith('/capabilities')) data = { ok: true, pages: Object.fromEntries(pages.map(p => [p, true])), actions: {} };
    if (p.endsWith('/drugs-labs-kpis')) {
      if (failFamily) return route.fulfill({ status: 503, json: { detail: 'synthetic unavailable' } });
      const family = (id: string) => ({ id, cases: 10, pct: 10,
        tiles: [{ id: 'any', title_ru: 'МО с замечаниями', cases: 10, pct: 10, denominator: 'total_cases', denominator_n: 100 }],
        by_code: [{ code: id === 'lab' ? 'B_lab_unused_in_dx' : 'C_ddi', title_ru: 'Тестовое замечание', cases: 10, pct: 10 }],
        by_specialty: [], by_doctor: [] });
      data = { ok: true, families: { lab: family('lab'), drug: family('drug') }, denominators: { total_cases: 100, lab_coverage_available: false } };
    }
    if (p.endsWith('/daily-report')) data = { ok: true, date: '2026-08-02', data_through: '2026-08-02', attention: { n_evaluated: 100 }, actions: [], data_completeness: {} };
    if (p.endsWith('/month-report')) data = { ok: true, available: false, reason: 'Нет синтетических данных месяца', facets: {} };
    if (p.endsWith('/score-dashboard')) {
      const bands = { ok: { n: 70 }, weak: { n: 20 }, bad: { n: 10 }, na: { n: 0 } };
      data = { ok: true, available: true, window: { date_from: '2026-08-01', date_to: '2026-08-02' },
        zones: Object.fromEntries(['zone1', 'zone2a', 'zone2b'].map(key => [key, { avg_pct: 78, bands }])),
        reg55: { available: true, avg_pct: 82, band_share: { compliant_min: { n: 70 }, compliant_measures: { n: 20 }, noncompliant: { n: 10 }, unscored: { n: 0 } } }, trends: [] };
    }
    await route.fulfill({ json: data });
  });
  return { requests, problems };
}

for (const name of pages) {
  test(`МО: ${name} загружается с настоящими ассетами и CSP`, async ({ page }) => {
    const state = await mockMo(page);
    const response = await page.goto(`/methodist/mo?page=${name}`);
    expect(response?.status()).toBe(200);
    expect(response?.headers()['content-security-policy']).toContain("object-src 'none'");
    await expect(page.locator(`#page-${name}`)).toBeVisible();
    await expect(page.locator('#token-gate')).not.toBeVisible();
    expect(state.problems).toEqual([]);
  });
}

test('МО: ECharts показывает числа API и период передаётся в запрос', async ({ page }) => {
  const state = await mockMo(page);
  await page.goto('/methodist/mo?page=yesterday&period=month');
  const rings = page.locator('#yesterday-score-rings .score-ring-chart');
  await expect(rings).toHaveCount(4);
  await expect(page.locator('#yesterday-score-rings canvas')).toHaveCount(4);
  await expect(page.locator('#yesterday-score-rings .score-ring-meta')).toHaveText(['78%', '78%', '78%', '82%']);
  const values = await rings.evaluateAll(nodes => nodes.map(node => {
    const charts = (window as unknown as { echarts: { getInstanceByDom(el: Element): { getOption(): { series: { data: { value: number }[] }[] } } } }).echarts;
    return charts.getInstanceByDom(node).getOption().series[0].data.map(item => item.value);
  }));
  expect(values).toEqual(Array.from({ length: 4 }, () => [70, 20, 10]));
  expect(state.requests.find(url => url.pathname.endsWith('/score-dashboard'))?.searchParams.get('period')).toBe('month');
  expect(state.problems).toEqual([]);
});

test('МО: семейство, процент и drill сохраняют срез на мобильном', async ({ page }) => {
  const state = await mockMo(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/methodist/mo?page=labs&period=month');
  await expect(page.locator('#labs-kpis .kpi-value')).toHaveText('10%');
  await expect(page.locator('#labs-coverage')).toContainText('всех 100 МО периода');
  await page.locator('#labs-kpis button').click();
  await expect(page).toHaveURL(/page=documents/);
  await expect.poll(() => state.requests.filter(url => url.pathname.endsWith('/cases')).at(-1)?.searchParams.get('finding_family')).toBe('lab');
  expect(state.requests.filter(url => url.pathname.endsWith('/cases')).at(-1)?.searchParams.get('period')).toBe('month');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(state.problems).toEqual([]);
});

test('МО: отказ API не отображается как нулевое число замечаний', async ({ page }) => {
  await mockMo(page, true);
  await page.goto('/methodist/mo?page=labs');
  await expect(page.locator('#global-error')).toBeVisible();
  await expect(page.locator('#global-error')).toHaveText('Не удалось загрузить сводку анализов.');
  await expect(page.locator('#labs-kpis .kpi-value')).toHaveCount(0);
});

test('МО: задержанный ответ старого среза не перерисовывает новый', async ({ page }) => {
  const problems: string[] = [];
  let familyCalls = 0;
  page.on('pageerror', error => problems.push(error.message));
  await page.addInitScript(() => localStorage.setItem('protocol_methodist_token', 'synthetic-local-only'));
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname;
    if (path.endsWith('/capabilities')) {
      await route.fulfill({ json: { ok: true, pages: { labs: true }, actions: {} } });
      return;
    }
    if (!path.endsWith('/drugs-labs-kpis')) {
      await route.fulfill({ json: { ok: true, items: [], rows: [], facets: {} } });
      return;
    }
    familyCalls += 1;
    const call = familyCalls;
    if (call === 1) await page.waitForTimeout(700);
    const pct = call === 1 ? 11 : 77;
    const family = (id: string) => ({
      id, cases: pct, pct,
      tiles: [{ id: 'any', title_ru: 'МО с замечаниями', cases: pct, pct, denominator: 'total_cases', denominator_n: 100 }],
      by_code: [], by_specialty: [], by_doctor: []
    });
    try {
      await route.fulfill({
        json: {
          ok: true,
          families: { lab: family('lab'), drug: family('drug') },
          denominators: { total_cases: 100, lab_coverage_available: false }
        }
      });
    } catch {
      // AbortController штатно закрывает первый request при смене среза.
    }
  });

  await page.goto('/methodist/mo?page=labs&period=7d');
  await expect.poll(() => familyCalls).toBe(1);
  await page.locator('#period').evaluate((select: HTMLSelectElement) => {
    select.value = 'month';
    select.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await expect(page.locator('#labs-kpis .kpi-value')).toHaveText('77%');
  await page.waitForTimeout(900);
  await expect(page.locator('#labs-kpis .kpi-value')).toHaveText('77%');
  await expect(page.locator('#global-error')).toBeHidden();
  expect(familyCalls).toBe(2);
  expect(problems).toEqual([]);
});
