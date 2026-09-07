import { expect, test, type Page, type Route } from '@playwright/test';

const capabilities = {
  ok: true,
  pages: {
    yesterday: true, overview: true, queue: true, documents: true,
    doctors: true, medications: true, labs: true, reports: true,
    'kp-sync': true, 'rceth-sync': true, settings: true
  },
  actions: {}
};

const cases = [
  { id: 'case-a', visit_id: 'case-a', patient_id: 'patient-a', date: '2026-01-01',
    doctor: 'Врач А', specialty: 'Терапевт', diagnosis: 'Синтетический диагноз А' },
  { id: 'case-b', visit_id: 'case-b', patient_id: 'patient-b', date: '2026-01-02',
    doctor: 'Врач Б', specialty: 'Терапевт', diagnosis: 'Синтетический диагноз Б' }
];

function detail(id: string) {
  const row = cases.find(item => item.id === id) || cases[0];
  return {
    ok: true,
    record: { ...row, overall_pct: 0 },
    document: {
      source_format: 'secure_csv',
      clinical: {
        complaints: `Только синтетический текст ${id}. `.repeat(80)
      }
    },
    assessment: {
      contract_version: 1, status: 'completed', value: 0,
      confirmed_value: 0, protocol: { applicability_status: 'not_evaluated' }
    },
    zones: { ok: false },
    findings: [],
    review_packs: [],
    events: [],
    medication_normative_cards: { cards: [], shadow: true, primary: false }
  };
}

async function baseSetup(page: Page, handler?: (route: Route, url: URL) => Promise<boolean>) {
  await page.addInitScript(() => localStorage.setItem('protocol_methodist_token', 'synthetic-local-only'));
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url());
    if (handler && await handler(route, url)) return;
    if (url.pathname.endsWith('/capabilities')) {
      await route.fulfill({ json: capabilities });
      return;
    }
    if (url.pathname.endsWith('/cases')) {
      await route.fulfill({ json: { ok: true, items: cases, total: cases.length, facets: {} } });
      return;
    }
    const match = url.pathname.match(/\/cases\/([^/]+)$/);
    if (match) {
      await route.fulfill({ json: detail(decodeURIComponent(match[1])) });
      return;
    }
    await route.fulfill({ json: { ok: true, items: [], rows: [], facets: {} } });
  });
}

test('E03: late previous case response cannot replace current drawer', async ({ page }) => {
  await baseSetup(page, async (route, url) => {
    if (url.pathname.endsWith('/cases/case-a')) {
      await page.waitForTimeout(700);
      try { await route.fulfill({ json: detail('case-a') }); } catch {}
      return true;
    }
    if (url.pathname.endsWith('/cases/case-b')) {
      await route.fulfill({ json: detail('case-b') });
      return true;
    }
    return false;
  });
  await page.goto('/methodist/mo?page=documents');
  await expect(page.locator('#document-rows tr[data-case]')).toHaveCount(2);
  await page.locator('#document-rows tr[data-case]').nth(0).click();
  await page.locator('#document-rows tr[data-case]').nth(1).evaluate((row: HTMLElement) => row.click());
  await expect(page.locator('#drawer-subtitle')).toContainText('case-b');
  await page.waitForTimeout(900);
  await expect(page.locator('#drawer-subtitle')).toContainText('case-b');
  await expect(page.locator('#drawer-body')).not.toContainText('case-a');
});

test('E21: protocol widget 500 does not hide the clinical document', async ({ page }) => {
  await baseSetup(page, async (route, url) => {
    if (url.pathname.endsWith('/cases/case-a/protocol-suggest')) {
      await route.fulfill({ status: 500, json: { detail: 'synthetic failure' } });
      return true;
    }
    return false;
  });
  await page.goto('/methodist/mo?page=documents');
  await page.locator('#document-rows tr[data-case]').first().click();
  await expect(page.locator('#case-clinical-pane')).toContainText('Только синтетический текст');
  await expect(page.locator('#protocol-suggest-host')).toContainText('Не удалось подобрать протоколы');
  await expect(page.locator('#case-clinical-pane')).toBeVisible();
});

test('E22: keyboard, viewport matrix, zoom and long text stay usable', async ({ page }) => {
  await baseSetup(page);
  for (const width of [320, 360, 768, 1024, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    await page.goto('/methodist/mo?page=documents');
    const navLabels = await page.locator('#app-nav .nav-button:visible').evaluateAll(
      nodes => nodes.map(node => node.getAttribute('aria-label')).filter(Boolean)
    );
    expect(navLabels.length).toBeGreaterThan(5);
    const filters = page.locator('#filters-panel > summary');
    await filters.focus();
    await page.keyboard.press('Enter');
    await expect(page.locator('#filters-panel')).toHaveAttribute('open', '');
    const box = await page.locator('#filters-panel .toolbar-panel').boundingBox();
    expect(box).not.toBeNull();
    expect((box?.x || 0) + (box?.width || 0)).toBeLessThanOrEqual(width + 1);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
  // 360 CSS px at 200% browser zoom has an effective layout width of 180 px.
  await page.setViewportSize({ width: 180, height: 450 });
  await page.goto('/methodist/mo?page=documents');
  await page.locator('#filters-panel').evaluate((panel: HTMLDetailsElement) => { panel.open = false; });
  await page.locator('#document-rows tr[data-case]').first().focus();
  await page.keyboard.press('Enter');
  await page.locator('[data-case-tab="review"]').focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('#case-review-column')).toBeVisible();
  await page.locator('[data-case-tab="document"]').focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('#case-clinical-pane')).toContainText('Только синтетический текст');
  expect(await page.locator('#case-drawer').evaluate(
    drawer => drawer.scrollWidth <= drawer.clientWidth
  )).toBe(true);
});

