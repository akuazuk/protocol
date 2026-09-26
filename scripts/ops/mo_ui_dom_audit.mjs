#!/usr/bin/env node
/**
 * DOM-аудит кабинета МО Аналитики (план 2026-09-26, волна T; метод раздела 1).
 *
 * Проходит по всем экранам под сессией методиста в двух viewport и для каждого
 * пишет: высоту липкой шапки, число открытых/закрытых <details>, скрытые пункты
 * меню, карточки, экземпляры ECharts (серии/точки), таблицы (колонки/строки/
 * высота строки/горизонтальный overflow), элементы за правым краем, реальные
 * шрифты, `document.fonts`, тайминги /api/methodist/mo/* и ячейки с сырым «<br».
 * Скриншотов и текстов не сохраняет - только числа и короткие метки (без ФИО).
 *
 *   METHODIST_TOKEN=... node scripts/ops/mo_ui_dom_audit.mjs \
 *       --base https://protocol.kravira.by --out /tmp/dom-audit.json
 *
 *   node scripts/ops/mo_ui_dom_audit.mjs --compare before.json after.json
 *
 * Требует @playwright/test из package.json (npm ci) и chromium (npx playwright install chromium).
 */
import fs from "node:fs";
import { chromium } from "@playwright/test";

const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : def;
};

if (args.includes("--compare")) {
  const i = args.indexOf("--compare");
  process.exit(compare(args[i + 1], args[i + 2]));
}

const BASE = opt("--base", process.env.MO_AUDIT_BASE || "https://protocol.kravira.by");
const OUT = opt("--out", null);
const TOKEN = (process.env.METHODIST_TOKEN || "").trim();
const VIEWPORTS = (opt("--viewports", "1024x900,1440x1000")).split(",").map((v) => {
  const [w, h] = v.split("x").map(Number);
  return { width: w, height: h };
});
const PAGES = (opt("--pages", "yesterday,documents,mis,doctors,medications,labs,queue,reports,kp-sync,rceth-sync,settings")).split(",");
const WAIT_MS = Number(opt("--wait", "6000"));

if (!TOKEN) {
  console.error("METHODIST_TOKEN не задан");
  process.exit(2);
}

function auditInPage() {
  const vis = (el) => {
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    return r.width > 0 && r.height > 0 && cs.visibility !== "hidden" && cs.display !== "none";
  };
  const root = document.querySelector("main") || document.body;
  const sections = Array.from(root.querySelectorAll("section.page, section")).filter((s) => !s.hidden && vis(s));
  const out = { sections: [] };
  sections.forEach((sec) => {
    const cards = Array.from(sec.querySelectorAll(".card, article, .panel, .kpi, .tile")).filter(vis);
    const s = { id: sec.id, height: Math.round(sec.getBoundingClientRect().height), cards: [] };
    cards.forEach((c) => {
      const h = c.querySelector("h2, h3, .card-title, .kpi-label");
      const charts = Array.from(c.querySelectorAll("[_echarts_instance_]")).map((e) => {
        try {
          const inst = window.echarts.getInstanceByDom(e);
          const o = inst && inst.getOption();
          const series = (o && o.series) || [];
          return { series: series.length, points: series.reduce((a, sr) => a + ((sr.data || []).length), 0), w: e.clientWidth, h: e.clientHeight };
        } catch (err) { return { err: String(err) }; }
      });
      const tables = Array.from(c.querySelectorAll("table")).map((t) => {
        const tr = t.querySelector("tbody tr");
        return {
          cols: t.querySelectorAll("thead th").length,
          rows: t.querySelectorAll("tbody tr").length,
          rowH: tr ? Math.round(tr.getBoundingClientRect().height) : 0,
          overflowX: t.scrollWidth > (t.parentElement ? t.parentElement.clientWidth : t.clientWidth) + 2,
          nestedScroll: (() => { let p = t.parentElement; while (p && p !== root) { const cs = getComputedStyle(p); if (/auto|scroll/.test(cs.overflowY) && p.scrollHeight > p.clientHeight + 4) return true; p = p.parentElement; } return false; })(),
        };
      });
      const txt = (c.innerText || "").replace(/\s+/g, " ").trim();
      s.cards.push({
        title: h ? h.textContent.trim().slice(0, 50) : "(no title)",
        h: Math.round(c.getBoundingClientRect().height),
        charts, tables,
        emptyLike: /нет данных|пока нет|не найдено|загрузка|нет случаев|пусто|ошибка|error|422|500/i.test(txt.slice(0, 300)),
      });
    });
    out.sections.push(s);
  });
  const all = Array.from(document.querySelectorAll("*"));
  out.overflowRight = all.filter((e) => { const r = e.getBoundingClientRect(); return r.right > innerWidth + 2 && vis(e) && e.children.length === 0; }).length;
  out.detailsClosed = Array.from(document.querySelectorAll("details:not([open])")).filter((d) => vis(d) && !d.closest(".filter-pop, .toolbar-section, .nav-more-menu, .column-all")).length;
  out.detailsTotal = Array.from(document.querySelectorAll("details")).filter(vis).length;
  out.navHidden = document.querySelectorAll("#app-nav .nav-button[hidden]").length;
  out.navVisible = Array.from(document.querySelectorAll("#app-nav .nav-button")).filter(vis).length;
  out.navHasMore = !!document.querySelector("#nav-more");
  out.rawBrCells = Array.from(document.querySelectorAll("td, .clinical-field p, .case-dx")).filter((e) => /<br\s*\/?>/i.test(e.textContent || "")).length;
  out.gluedStatus = Array.from(document.querySelectorAll(".assessment-status-strip")).filter((e) => /[а-яё][А-ЯЁ]/.test((e.textContent || "").replace(/\s/g, ""))).length;
  out.fonts = Array.from(new Set(Array.from(document.querySelectorAll("h1,h2,h3,.kpi-value,.kpi-num,td,th,button,label")).slice(0, 300).map((e) => { const cs = getComputedStyle(e); return `${e.tagName} ${cs.fontFamily.split(",")[0]} ${cs.fontSize} ${cs.fontWeight}`; }))).slice(0, 30);
  out.webfonts = Array.from(document.fonts || []).filter((f) => f.status === "loaded").map((f) => `${f.family} ${f.weight}`).slice(0, 12);
  const header = document.querySelector('header, .context-bar, [aria-label="Поиск и глобальные фильтры"]');
  out.stickyHeaderH = header ? Math.round(header.getBoundingClientRect().height) : null;
  out.api = performance.getEntriesByType("resource").filter((r) => r.name.includes("/api/methodist/mo/")).map((r) => ({ path: new URL(r.name).pathname.replace(/\/\d{4,}/g, "/<id>"), ms: Math.round(r.duration), status: r.responseStatus || null }));
  out.echartsTotal = document.querySelectorAll("[_echarts_instance_]").length;
  return out;
}

async function run() {
  const browser = await chromium.launch();
  const doc = { taken_at: new Date().toISOString(), base: BASE, viewports: {} };
  for (const vp of VIEWPORTS) {
    const key = `${vp.width}x${vp.height}`;
    doc.viewports[key] = {};
    const ctx = await browser.newContext({ viewport: vp, locale: "ru-RU" });
    await ctx.addInitScript((token) => {
      try { localStorage.setItem("protocol_methodist_token", token); } catch (e) { /* ignore */ }
    }, TOKEN);
    const page = await ctx.newPage();
    const errors = [];
    page.on("pageerror", (e) => errors.push(String(e.message || e).slice(0, 160)));
    page.on("console", (m) => { if (m.type() === "error") errors.push(m.text().slice(0, 160)); });
    for (const pid of PAGES) {
      await page.goto(`${BASE}/methodist/mo?page=${pid}`, { waitUntil: "domcontentloaded" });
      await page.waitForTimeout(WAIT_MS);
      const res = await page.evaluate(auditInPage);
      res.errors = errors.splice(0);
      doc.viewports[key][pid] = res;
      const closed = res.detailsClosed, ch = res.echartsTotal, ov = res.overflowRight;
      console.error(`${key} ${pid.padEnd(11)} header ${String(res.stickyHeaderH).padStart(3)}px charts ${ch} closedDetails ${closed} overflow ${ov} navHidden ${res.navHidden} more ${res.navHasMore} rawBr ${res.rawBrCells} webfonts ${res.webfonts.length}`);
    }
    // Разбор первого случая из «Найти МО».
    try {
      await page.goto(`${BASE}/methodist/mo?page=documents`, { waitUntil: "domcontentloaded" });
      await page.waitForTimeout(WAIT_MS);
      const row = page.locator("#document-rows tr[data-case]").first();
      if (await row.count()) {
        await row.click();
        await page.waitForTimeout(WAIT_MS);
        const res = await page.evaluate(auditInPage);
        res.errors = errors.splice(0);
        doc.viewports[key].case = res;
        console.error(`${key} case        closedDetails ${res.detailsClosed}/${res.detailsTotal} glued ${res.gluedStatus} rawBr ${res.rawBrCells} charts ${res.echartsTotal}`);
      }
    } catch (err) {
      doc.viewports[key].case = { error: String(err).slice(0, 200) };
    }
    await ctx.close();
  }
  await browser.close();
  const text = JSON.stringify(doc, null, 2);
  if (OUT) { fs.writeFileSync(OUT, text); console.error(`saved ${OUT}`); } else { console.log(text); }
}

function summarize(doc) {
  const rows = [];
  for (const [vp, pages] of Object.entries(doc.viewports)) {
    for (const [pid, r] of Object.entries(pages)) {
      if (!r || r.error) continue;
      rows.push({ key: `${vp}/${pid}`, header: r.stickyHeaderH, charts: r.echartsTotal, closed: r.detailsClosed, overflow: r.overflowRight, navHidden: r.navHidden, rawBr: r.rawBrCells, errors: (r.errors || []).length, webfonts: (r.webfonts || []).length });
    }
  }
  return rows;
}

function compare(beforePath, afterPath) {
  const before = summarize(JSON.parse(fs.readFileSync(beforePath, "utf8")));
  const after = summarize(JSON.parse(fs.readFileSync(afterPath, "utf8")));
  const idx = Object.fromEntries(before.map((r) => [r.key, r]));
  let fail = 0;
  for (const r of after) {
    const b = idx[r.key];
    const worse = b && (r.closed > b.closed || r.overflow > b.overflow || r.navHidden > b.navHidden || r.rawBr > b.rawBr || r.errors > b.errors || r.charts < b.charts);
    if (worse) fail += 1;
    console.log(`${r.key.padEnd(24)} header ${String(r.header).padStart(3)} charts ${r.charts} closed ${r.closed} overflow ${r.overflow} rawBr ${r.rawBr} errors ${r.errors} webfonts ${r.webfonts}` + (b ? `  (было: charts ${b.charts} closed ${b.closed} overflow ${b.overflow} rawBr ${b.rawBr} errors ${b.errors})` : "  (new)") + (worse ? "  REGRESS" : ""));
  }
  return fail ? 1 : 0;
}

run().catch((err) => { console.error(err); process.exit(1); });
