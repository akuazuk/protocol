import { test, expect, type Page } from '@playwright/test';

// Real MO HTML/assets/CSP; synthetic API only. No clinical records or model calls.
const pages = ['yesterday', 'queue', 'documents', 'doctors', 'medications', 'labs', 'reports', 'kp-sync', 'rceth-sync', 'settings'];

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
    if (p.endsWith('/labs-dashboard')) {
      if (failFamily) return route.fulfill({ status: 503, json: { detail: 'synthetic unavailable' } });
      data = {
        ok: true, available: true, total_cases: 100,
        tiles: [
          { id: 'unused', label: 'Анализы не учтены', n: 10, n_cases: 10, pct: 10, tone: 'rose', codes: ['B_lab_unused_in_dx'] }
        ],
        window: { available: true, has: 40, none: 60, unused: 10, accounted: 30 },
        unused_tests: [{ label: 'ОАК', n: 5, n_cases: 4 }],
        abnormal_specialty: [{ specialty: 'Терапия', n_cases: 50, n: 5, pct: 10 }],
        trend: [{ week: '2026-W31', n: 3, unused: 2, abnormal: 1, present_not_in_mo: 0, exams_gap: 0, ordered: 0 }],
        trend_tiles: [{ id: 'unused', label: 'Анализы не учтены' }],
        coverage_months: { available: true, from: '2025-12-01', items: [{ month: '2026-08', n: 100, has: 40, pct: 40 }] }
      };
    }
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
    if (p.endsWith('/cases/summary')) {
      data = {
        ok: true, available: true, n: 12,
        grades: {
          totals: { good: 5, fair: 4, poor: 3, important: 0, critical: 0, na: 0 },
          buckets: [
            { id: 'good', label: 'Хорошо', n: 5 },
            { id: 'fair', label: 'С замечанием', n: 4 },
            { id: 'poor', label: 'Слабо', n: 3 }
          ]
        },
        specialties: [{ value: 'Терапевт', n: 7 }, { value: 'Кардиолог', n: 5 }],
        weeks: [{ week: '2026-W31', date_from: '2026-08-01', date_to: '2026-08-02', n: 12 }]
      };
    }
    if (p.endsWith('/doctors-dashboard')) {
      data = {
        ok: true, available: true, rank_n: 20,
        ranking: [{ key: 'a', label: 'Врач А', specialty: 'Терапия', n: 40, enough: true, zone1_bad_pct: 20, zone2a_bad_pct: 10, zone2b_bad_pct: 30 }],
        heatmap: { zones: [{ id: 'zone1', label: 'Оформление' }, { id: 'zone2a', label: 'Диагноз' }, { id: 'zone2b', label: 'План' }],
          rows: [{ key: 'a', label: 'Врач А', n: 40, cells: [
            { zone: 'zone1', n: 40, bad: 8, bad_pct: 20, suppressed: false },
            { zone: 'zone2a', n: 40, bad: 4, bad_pct: 10, suppressed: false },
            { zone: 'zone2b', n: 40, bad: 12, bad_pct: 30, suppressed: false }
          ] }] },
        scatter: [{ key: 'a', label: 'Врач А', n: 40, bad_pct: 20, enough: true }],
        selected: { key: 'a', label: 'Врач А', specialty: 'Терапия', n: 40, enough: true,
          radar: [{ id: 'zone1', label: 'Оформление', ok_pct: 80, bad_pct: 20 }],
          findings_top: [], chapters: [], trend: [{ week: '2026-W31', n: 10, zone1_avg: 70, zone2a_avg: 80, zone2b_avg: 60 }],
          specialty_median: [{ week: '2026-W31', zone1_avg: 75, zone2a_avg: 82, zone2b_avg: 58 }] }
      };
    }
    if (p.endsWith('/score-dashboard') || p.endsWith('/overview-dashboard')) {
      // F1: Обзор берёт один /overview-dashboard; /score-dashboard остаётся fallback для старого образа.
      const bands = { ok: { n: 70 }, weak: { n: 20 }, bad: { n: 10 }, na: { n: 0 } };
      data = { ok: true, available: true, granularity: 'day', window: { date_from: '2026-08-01', date_to: '2026-08-02' },
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

test('МО: page=overview открывает Обзор, не дубль Период', async ({ page }) => {
  const state = await mockMo(page);
  const response = await page.goto('/methodist/mo?page=overview');
  expect(response?.status()).toBe(200);
  await expect(page.locator('#page-yesterday')).toBeVisible();
  await expect(page.locator('#page-overview')).toBeHidden();
  expect(state.problems).toEqual([]);
});

test('МО: ECharts показывает числа API и период передаётся в запрос', async ({ page }) => {
  const state = await mockMo(page);
  await page.goto('/methodist/mo?page=yesterday&period=month');
  const rings = page.locator('#yesterday-score-rings .score-ring-chart');
  await expect(rings).toHaveCount(3);
  await expect(page.locator('#yesterday-score-rings canvas')).toHaveCount(3);
  // Компактные кольца: общая легенда из 4 сегментов кольца (хорошо / слабо / важно / нет оценки).
  await expect(page.locator('#yesterday-score-rings .score-grade-legend__item')).toHaveCount(4);
  // Центр кольца - доля «хорошо» (70 из 100 в моке), слово шкалы - в подписи под кольцом.
  await expect(page.locator('#yesterday-score-rings .score-ring-meta')).toHaveText(['70%', '70%', '70%']);
  await expect(page.locator('#yesterday-score-rings .score-ring-denominator').first()).toContainText('чаще всего: хорошо');
  const values = await rings.evaluateAll(nodes => nodes.map(node => {
    const charts = (window as unknown as { echarts: { getInstanceByDom(el: Element): { getOption(): { series: { data: { value: number }[] }[] } } } }).echarts;
    return charts.getInstanceByDom(node).getOption().series[0].data.map(item => item.value);
  }));
  expect(values).toEqual(Array.from({ length: 3 }, () => [70, 20, 10]));
  expect(state.requests.find(url => url.pathname.endsWith('/overview-dashboard'))?.searchParams.get('period')).toBe('month');
  expect(state.problems).toEqual([]);
});

test('МО: Найти МО показывает сводку выборки', async ({ page }) => {
  const state = await mockMo(page);
  await page.goto('/methodist/mo?page=documents');
  await expect(page.locator('#page-documents')).toBeVisible();
  await expect(page.locator('#cases-summary')).toBeVisible();
  await expect(page.locator('#cases-summary-grades .cases-summary-bar')).toHaveCount(3);
  await expect(page.locator('#cases-summary-specialties .cases-summary-bar')).toHaveCount(2);
  await expect(page.locator('#cases-summary-weeks canvas')).toHaveCount(1);
  expect(state.requests.some(url => url.pathname.endsWith('/cases/summary'))).toBe(true);
  expect(state.problems).toEqual([]);
});

test('МО: семейство, процент и drill сохраняют срез на мобильном', async ({ page }) => {
  const state = await mockMo(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/methodist/mo?page=labs&period=month');
  await expect(page.locator('#labs-kpis .kpi-value')).toHaveText('10');
  await expect(page.locator('#labs-coverage')).toContainText('не учтена 10');
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
    if (!path.endsWith('/labs-dashboard')) {
      await route.fulfill({ json: { ok: true, items: [], rows: [], facets: {} } });
      return;
    }
    familyCalls += 1;
    const call = familyCalls;
    if (call === 1) await page.waitForTimeout(700);
    const nCases = call === 1 ? 11 : 77;
    try {
      await route.fulfill({
        json: {
          ok: true,
          available: true,
          total_cases: 100,
          tiles: [{ id: 'unused', label: 'Анализы не учтены', n: nCases, n_cases: nCases, pct: nCases, tone: 'rose', codes: [] }],
          window: { available: true, has: 40, none: 60, unused: nCases, accounted: 30 },
          unused_tests: [],
          abnormal_specialty: [],
          trend: [],
          trend_tiles: [],
          coverage_months: { available: false, items: [] }
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
  await expect(page.locator('#labs-kpis .kpi-value')).toHaveText('77');
  await page.waitForTimeout(900);
  await expect(page.locator('#labs-kpis .kpi-value')).toHaveText('77');
  await expect(page.locator('#global-error')).toBeHidden();
  expect(familyCalls).toBe(2);
  expect(problems).toEqual([]);
});
