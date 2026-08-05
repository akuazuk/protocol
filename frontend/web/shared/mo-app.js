  (function (MO) {
    "use strict";
    // Compatibility marker for older route tests: MIS · качество КЗ
    var TOKEN_KEY = "protocol_methodist_token";
    var VIEWS_KEY = "protocol_mo_saved_views";
    var THEME_KEY = "protocol_mo_theme";
    var DENSITY_KEY = "protocol_mo_density";
    var API_ROOT = MO.api.API_ROOT;
    var LEGACY_ROOT = MO.api.LEGACY_ROOT;
    var request = MO.api.request;
    var token = MO.api.token;
    var headers = MO.api.headers;
    var state = {
      page: "overview", period: "month", compare: "previous", methodology: "v3", pageNo: 1, dateFrom: "", dateTo: "", search: "", findingCode: "", rubricCriterion: "",
      sortBy: "date", sortDir: "desc",
      selected: { months: [], branches: [], specialties: [], doctors: [], document_types: [], statuses: [] },
      data: {}, facets: {}, trigger: null, openCaseId: "", cabinetDoctorKey: "",
      drillTrail: [], drillSnapshot: null,
      columnVisible: { documents: [], queue: [] }, columnsPanelOpen: false
    };
    var PAGE_TITLES = {
      overview: "Обзор МО", yesterday: "Отчёт за вчера", queue: "Очередь разбора",
      documents: "Все случаи", doctors: "Врачи", specialties: "Специальности",
      diagnoses: "Диагнозы и МКБ", safety: "Безопасность", "data-quality": "Качество данных",
      "doctor-cabinet": "Кабинет врача", "access-log": "Журнал доступа",
      reports: "Отчёты", settings: "Настройки"
    };
    var FILTER_LABELS = {
      months: "Месяц", branches: "Филиал", specialties: "Специальность", doctors: "Врач",
      document_types: "Тип документа", statuses: "Статус"
    };
    var API_FILTER_KEYS = {
      months: "periods", branches: "filials", specialties: "specializations", doctors: "doctors",
      document_types: "document_kinds", statuses: "statuses"
    };
    function $(id) { return document.getElementById(id); }
    function preference(key, fallback) {
      try { return localStorage.getItem(key) || fallback; } catch (error) { return fallback; }
    }
    function showToast(message) {
      var toast = document.createElement("div");
      toast.className = "toast";
      toast.textContent = message;
      $("toast-region").appendChild(toast);
      $("announcer").textContent = message;
      window.setTimeout(function () { toast.remove(); }, 3500);
    }
    function applyPreferences() {
      var theme = preference(THEME_KEY, "");
      var density = preference(DENSITY_KEY, "comfortable");
      if (theme) document.documentElement.dataset.theme = theme;
      else delete document.documentElement.dataset.theme;
      document.documentElement.dataset.density = density;
      $("density").value = density;
      var dark = theme === "dark" || (!theme && window.matchMedia("(prefers-color-scheme: dark)").matches);
      $("theme-toggle").setAttribute("aria-pressed", dark ? "true" : "false");
      $("theme-toggle").setAttribute("aria-label", dark ? "Включить светлую тему" : "Включить тёмную тему");
    }
    function esc(value) {
      return String(value == null ? "" : value).replace(/&/g, "&amp;").replace(/</g, "&lt;")
        .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    }
    function downloadBlob(blob, filename) {
      var url = URL.createObjectURL(blob), link = document.createElement("a");
      link.href = url; link.download = filename; document.body.appendChild(link); link.click();
      link.remove(); setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
    }
    function downloadJson(data, filename) {
      downloadBlob(new Blob([JSON.stringify(data, null, 2)], { type: "application/json;charset=utf-8" }), filename);
    }
    async function openPdfWithToken(path, options) {
      options = options || {};
      var targetWindow = options.targetWindow || null;
      var preferredName = options.preferredName || "mo-case.pdf";
      var response = await fetch(path, { headers: headers() });
      if (response.status === 401 || response.status === 403) {
        if (targetWindow && !targetWindow.closed) targetWindow.close();
        throw new Error("Требуется вход методиста. Обновите токен и повторите.");
      }
      if (!response.ok) {
        if (targetWindow && !targetWindow.closed) targetWindow.close();
        throw new Error("Не удалось открыть PDF МО.");
      }
      var type = (response.headers.get("content-type") || "").toLowerCase();
      if (type.indexOf("application/pdf") >= 0) {
        var pdfBlob = await response.blob();
        var pdfUrl = URL.createObjectURL(pdfBlob);
        if (targetWindow && !targetWindow.closed) {
          targetWindow.location.replace(pdfUrl);
        } else {
          var popup = window.open(pdfUrl, "_blank", "noopener");
          if (!popup) downloadBlob(pdfBlob, preferredName);
        }
        window.setTimeout(function () { URL.revokeObjectURL(pdfUrl); }, 30000);
        return;
      }
      var htmlText = await response.text();
      var doc = (targetWindow && !targetWindow.closed) ? targetWindow : window.open("", "_blank", "noopener");
      if (!doc) throw new Error("Браузер заблокировал всплывающее окно.");
      doc.document.open();
      doc.document.write(htmlText);
      doc.document.close();
    }
    function minskDateKey(dayOffset) {
      var parts = new Intl.DateTimeFormat("en-CA", {
        timeZone: "Europe/Minsk", year: "numeric", month: "2-digit", day: "2-digit"
      }).formatToParts(new Date()).reduce(function (result, part) {
        result[part.type] = part.value; return result;
      }, {});
      var calendar = new Date(Date.UTC(Number(parts.year), Number(parts.month) - 1, Number(parts.day) + (dayOffset || 0)));
      return calendar.getUTCFullYear() + "-" + String(calendar.getUTCMonth() + 1).padStart(2, "0") +
        "-" + String(calendar.getUTCDate()).padStart(2, "0");
    }
    async function exportCurrent(kind) {
      var filters = {};
      query().forEach(function (value, key) { filters[key] = value; });
      var response = await request("/exports", "/exports", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind: kind || "cases", filters: filters })
      });
      if (!response.ok) throw new Error("Не удалось подготовить выгрузку.");
      var job = await response.json();
      var file = await fetch(job.download_url, { headers: headers() });
      if (!file.ok) throw new Error("Не удалось скачать выгрузку.");
      downloadBlob(await file.blob(), "mo-export-" + job.job_id + ".json");
      $("announcer").textContent = "Выгрузка готова";
    }
    function query() {
      var q = new URLSearchParams();
      q.set("period", state.period); q.set("compare_period", state.compare);
      q.set("methodology", state.methodology);
      var today = minskDateKey(0);
      if (!state.selected.months.length) {
        if (state.period === "month") q.set("month", today.slice(0, 7));
        if (state.period === "yesterday") {
          q.set("date_from", minskDateKey(-1)); q.set("date_to", minskDateKey(-1));
        }
        if (state.period === "7d") {
          q.set("date_from", minskDateKey(-7)); q.set("date_to", minskDateKey(-1));
        }
        if (state.period === "custom" && state.dateFrom && state.dateTo) {
          q.set("date_from", state.dateFrom); q.set("date_to", state.dateTo);
        }
      }
      if (state.search) q.set("q", state.search);
      if (state.findingCode) q.set("finding_codes", state.findingCode);
      q.set("sort_by", state.sortBy);
      q.set("sort_dir", state.sortDir);
      Object.keys(state.selected).forEach(function (key) {
        if (state.selected[key].length) q.set(API_FILTER_KEYS[key] || key, state.selected[key].join(","));
      });
      if (state.selected.months.length) q.set("month", state.selected.months[0]);
      return q;
    }
    function normalizeSummary(raw) {
      raw = raw || {};
      var agg = raw.filtered_agg || raw.overview || raw;
      var contractKpi = raw.kpi || agg.kpi || {};
      return {
        raw: raw,
        n: contractKpi.source_records != null ? contractKpi.source_records : (agg.n || raw.n_ok || raw.n_cases || 0),
        evaluated: contractKpi.evaluated != null ? contractKpi.evaluated : (agg.n_evaluated || raw.n_ok || agg.n || 0),
        score: contractKpi.avg_score != null ? contractKpi.avg_score : (agg.avg_overall != null ? agg.avg_overall : raw.avg_overall_pct),
        attention: contractKpi.needs_attention != null ? contractKpi.needs_attention : (agg.n_bad || ((agg.status_distribution || {}).needs_review) || 0),
        attentionPct: contractKpi.needs_attention_pct != null ? contractKpi.needs_attention_pct : (agg.pct_bad || 0),
        critical: contractKpi.critical != null ? contractKpi.critical : (((agg.severity_totals || {}).P0) || raw.reg55_p0_defect_n || 0),
        coverage: agg.avg_coverage != null ? agg.avg_coverage : (raw.avg_coverage != null ? raw.avg_coverage : null),
        confidence: agg.avg_confidence != null ? agg.avg_confidence : (raw.avg_confidence != null ? raw.avg_confidence : null),
        specialties: raw.specialties || raw.by_specialty || agg.by_specialty || [],
        doctors: raw.doctors || raw.top_doctors || agg.by_doctor || [],
        branches: raw.filials || agg.by_branch || [],
        diagnoses: raw.by_chapter || agg.by_chapter || [],
        findings: agg.finding_types || raw.reg55_top_failed || [],
        generated: raw.generated_at || raw.data_freshness || raw.data_through || ""
      };
    }
    function kpi(label, value, meta, delta) {
      return '<article class="kpi"><div class="kpi-label">' + esc(label) + '</div><div class="kpi-value">' +
        esc(value == null ? "Нет данных" : value) + '</div><div class="kpi-meta">' +
        (delta ? '<span class="delta' + (String(delta).charAt(0) === "-" ? " down" : "") + '">' + esc(delta) + '</span> · ' : "") +
        esc(meta || "по выбранному периоду") + "</div></article>";
    }
    function score(value) {
      var numeric = Number(value);
      return value == null || value === "" || !Number.isFinite(numeric) ? "Нет данных" : Math.round(numeric) + "%";
    }
    function scoreLabel(value, reason) {
      if (value != null && value !== "") return Math.round(Number(value)) + "%";
      return reason || "Нет оценки";
    }
    function firstNumeric(values) {
      for (var i = 0; i < values.length; i++) {
        var value = values[i];
        if (value == null || value === "") continue;
        var num = Number(value);
        if (!Number.isNaN(num)) return num;
      }
      return null;
    }
    function looksLikeOpaqueHash(value) {
      var text = String(value == null ? "" : value).trim();
      return /^(?:[a-f0-9]{32}|[a-f0-9]{40}|[a-f0-9]{64})$/i.test(text);
    }
    function isIcdCode(value) {
      var text = String(value == null ? "" : value).trim();
      return /^[A-Za-zА-Яа-я]\d{2}(?:\.\d{1,2})?$/.test(text);
    }
    function normalizeDiagnosis(row) {
      var choices = [row.diagnosis_short, row.diagnosis, row.diagnosis_label, row.mkb_code_main, row.diagnosis_code];
      for (var i = 0; i < choices.length; i++) {
        var text = String(choices[i] == null ? "" : choices[i]).trim();
        if (!text || looksLikeOpaqueHash(text)) continue;
        if (i >= 3 && !isIcdCode(text)) continue;
        return text;
      }
      return "Не указан";
    }
    function deriveCoverage(detail, record, axes) {
      var value = firstNumeric([detail.coverage_pct, record.coverage_pct, record.coverage, record.deep_coverage_pct]);
      if (value != null) return { value: value, estimated: false };
      var axisKeys = ["documentation", "clinical_concordance", "safety", "regulatory"];
      var present = 0;
      axisKeys.forEach(function (key) {
        if (axes[key] != null || record["axis_" + key] != null) present += 1;
      });
      if (present > 0) return { value: Math.round((100 * present / axisKeys.length) * 10) / 10, estimated: true };
      if (String(record.parse_ok || "") === "1") return { value: 100, estimated: true };
      return { value: null, estimated: false };
    }
    function deriveConfidence(detail, record, axes) {
      var value = firstNumeric([detail.confidence_pct, record.confidence_pct, record.confidence, record.deep_confidence_pct]);
      if (value != null) return { value: value, estimated: false };
      var hasAxes = ["documentation", "clinical_concordance", "safety", "regulatory"].some(function (key) {
        return axes[key] != null || record["axis_" + key] != null;
      });
      var parseOk = String(record.parse_ok || "") === "1";
      var mismatch = String(record.date_mismatch || "0") === "1";
      if (parseOk && mismatch) return { value: 75, estimated: true };
      if (parseOk) return { value: 90, estimated: true };
      if (hasAxes) return { value: 55, estimated: true };
      return { value: null, estimated: false };
    }
    function bar(name, value, suffix) {
      var numeric = Number(value), available = value != null && value !== "" && Number.isFinite(numeric);
      var n = available ? Math.max(0, Math.min(100, numeric)) : 0;
      var cls = n < 60 ? " bad" : n < 75 ? " warn" : "";
      return '<div class="bar"><div class="bar-name" title="' + esc(name) + '">' + esc(name) +
        '</div><div class="track"><div class="fill' + (available ? cls : " unavailable") + '" style="width:' + n + '%"></div></div><b>' +
        esc(suffix == null ? (available ? Math.round(n) + "%" : "Нет данных") : suffix) + "</b></div>";
    }
    function notice(title, text, tone) {
      return '<div style="padding:10px 0;border-bottom:1px solid var(--line)"><span class="status ' +
        esc(tone || "review") + '">' + esc(title) + '</span><div style="margin-top:6px;font-size:13px">' + esc(text) + "</div></div>";
    }
    function showError(message) {
      $("global-error").textContent = message; $("global-error").hidden = false;
      showToast(message);
    }
    function setAuth(show, message) {
      $("auth").hidden = !show; $("dashboard").hidden = show;
      if (message) $("auth-error").textContent = message;
    }
    function values(list, keys) {
      return (list || []).map(function (item) {
        if (typeof item === "string") return { value: item, label: item, count: null };
        var value = "";
        for (var i = 0; i < keys.length; i++) if (item[keys[i]]) { value = item[keys[i]]; break; }
        return { value: String(value), label: String(value), count: item.n == null ? item.count : item.n };
      }).filter(function (x) { return x.value; });
    }
    function monthValues() {
      var out = [], cursor = new Date();
      cursor.setDate(1);
      for (var i = 0; i < 24; i++) {
        var value = cursor.getFullYear() + "-" + String(cursor.getMonth() + 1).padStart(2, "0");
        out.push({ value: value, label: cursor.toLocaleDateString("ru-RU", { month: "long", year: "numeric" }), count: null });
        cursor.setMonth(cursor.getMonth() - 1);
      }
      return out;
    }
    function buildFacets(summary, rawFacets) {
      rawFacets = rawFacets || {};
      state.facets = {
        months: monthValues(),
        branches: values(rawFacets.branches || rawFacets.filials || summary.branches, ["value","filial","branch"]),
        specialties: values(rawFacets.specialties || summary.specialties, ["value","specialization","specialty"]),
        doctors: values(rawFacets.doctors || summary.doctors, ["value","doctor_fio","doctor"]),
        document_types: values(rawFacets.document_types || rawFacets.document_kinds || rawFacets.kz_kind || ["medical_exam","consultation","certificate","diagnostic"], ["value"]),
        statuses: values(rawFacets.statuses || ["Хорошо","Требует внимания","Критично","Недостаточно данных"], ["value"])
      };
      document.querySelectorAll(".filter-pop").forEach(renderFilter);
      updateFilterSummary();
    }
    function appliedFilterCount() {
      return Object.keys(state.selected).reduce(function (total, key) {
        return total + (state.selected[key] || []).length;
      }, 0);
    }
    function updateFilterSummary() {
      var count = appliedFilterCount();
      var total = $("filter-total");
      if (total) total.textContent = count ? "Выбрано: " + count : "Без дополнительных фильтров";
      var clearSearch = $("case-search-clear");
      if (clearSearch) clearSearch.hidden = !String(state.search || $("case-search").value || "").trim();
    }
    function clearCaseSearch() {
      state.search = "";
      $("case-search").value = "";
      $("search-suggestions").hidden = true;
      $("case-search").setAttribute("aria-expanded", "false");
      filtersChanged();
      $("case-search").focus();
    }
    function closeOtherFilterMenus(current) {
      document.querySelectorAll(".filter-pop[open]").forEach(function (details) {
        if (details !== current) details.open = false;
      });
    }
    function renderFilter(details) {
      var key = details.getAttribute("data-filter");
      var list = state.facets[key] || [];
      var selected = (state.selected[key] || []).slice();
      var draft = selected.slice();
      details.querySelector("summary b").textContent = selected.length ? selected.length : "Все";
      details.querySelector(".filter-menu").innerHTML =
        '<input class="control" type="search" placeholder="Найти" aria-label="Поиск по фильтру">' +
        '<div class="filter-options">' + list.map(function (item) {
          return '<label class="filter-option" data-label="' + esc(item.label.toLowerCase()) + '"><input type="checkbox" value="' +
            esc(item.value) + '"' + (selected.indexOf(item.value) >= 0 ? " checked" : "") + '><span>' +
            esc(item.label) + '</span><span class="filter-count">' + esc(item.count == null ? "" : item.count) + "</span></label>";
        }).join("") + '</div><div class="filter-menu-actions"><span class="filter-selection">Без изменений</span>' +
        '<button class="button secondary compact" type="button" data-filter-clear>Очистить</button>' +
        '<button class="button compact" type="button" data-filter-apply>Применить</button></div>';
      var search = details.querySelector('input[type="search"]');
      var selection = details.querySelector(".filter-selection");
      var apply = details.querySelector("[data-filter-apply]");
      function renderDraftState() {
        var changed = JSON.stringify(draft) !== JSON.stringify(selected);
        selection.textContent = changed ? "Будет выбрано: " + draft.length : "Без изменений";
        apply.disabled = !changed;
        details.classList.toggle("has-pending", changed);
      }
      search.addEventListener("input", function () {
        var term = search.value.trim().toLowerCase();
        details.querySelectorAll(".filter-option").forEach(function (option) {
          option.hidden = option.getAttribute("data-label").indexOf(term) < 0;
        });
      });
      details.querySelectorAll('input[type="checkbox"]').forEach(function (input) {
        input.addEventListener("change", function () {
          var index = draft.indexOf(input.value);
          if (input.checked && index < 0) draft.push(input.value);
          if (!input.checked && index >= 0) draft.splice(index, 1);
          renderDraftState();
        });
      });
      details.querySelector("[data-filter-clear]").addEventListener("click", function () {
        draft = [];
        details.querySelectorAll('input[type="checkbox"]').forEach(function (input) {
          input.checked = false;
        });
        renderDraftState();
      });
      apply.addEventListener("click", function () {
        state.selected[key] = draft.slice();
        selected = draft.slice();
        details.querySelector("summary b").textContent = draft.length || "Все";
        details.classList.remove("has-pending");
        details.open = false;
        showToast(draft.length ? "Фильтр применён: " + FILTER_LABELS[key] : "Фильтр очищен: " + FILTER_LABELS[key]);
        filtersChanged();
      });
      details.ontoggle = function () {
        if (details.open) {
          closeOtherFilterMenus(details);
          window.setTimeout(function () { search.focus(); }, 0);
        } else if (JSON.stringify(draft) !== JSON.stringify(selected)) {
          draft = selected.slice();
          details.querySelectorAll('input[type="checkbox"]').forEach(function (input) {
            input.checked = draft.indexOf(input.value) >= 0;
          });
          renderDraftState();
        }
      };
      renderDraftState();
    }
    function selectionSnapshot() {
      return {
        period: state.period,
        compare: state.compare,
        dateFrom: state.dateFrom,
        dateTo: state.dateTo,
        search: state.search,
        findingCode: state.findingCode,
        selected: JSON.parse(JSON.stringify(state.selected || {}))
      };
    }
    function restoreSelection(snapshot) {
      if (!snapshot) return;
      state.period = snapshot.period || state.period;
      state.compare = snapshot.compare || state.compare;
      state.dateFrom = snapshot.dateFrom || "";
      state.dateTo = snapshot.dateTo || "";
      state.search = snapshot.search || "";
      state.findingCode = snapshot.findingCode || "";
      state.selected = JSON.parse(JSON.stringify(snapshot.selected || state.selected));
      $("period").value = state.period;
      $("compare").value = state.compare;
      $("date-from").value = state.dateFrom;
      $("date-to").value = state.dateTo;
      $("case-search").value = state.search;
      $("date-from-wrap").hidden = state.period !== "custom";
      $("date-to-wrap").hidden = state.period !== "custom";
      document.querySelectorAll(".filter-pop").forEach(renderFilter);
    }
    function renderAnalysisRail() {
      renderBreadcrumbs();
      var path = $("analysis-path"), note = $("analysis-note");
      if (!path || !note) return;
      if (!state.drillTrail.length) {
        path.innerHTML = "Переходите в графики и нажимайте на точки для drill-down.";
        note.textContent = "Текущие фильтры применяются ко всем экранам и таблицам.";
        return;
      }
      path.innerHTML = state.drillTrail.map(function (entry, index) {
        return '<span class="analysis-step">' + esc(entry.label || ("Шаг " + (index + 1))) +
          ' <button type="button" data-drill-index="' + index + '" aria-label="Открыть шаг">↗</button></span>';
      }).join("");
      note.textContent = "Страница: " + (PAGE_TITLES[state.page] || state.page) + " · шагов: " + state.drillTrail.length;
      path.querySelectorAll("[data-drill-index]").forEach(function (button) {
        button.addEventListener("click", function () {
          var index = Number(button.getAttribute("data-drill-index"));
          var entry = state.drillTrail[index];
          if (!entry) return;
          if (entry.apply) entry.apply();
        });
      });
    }
    function renderBreadcrumbs() {
      var root = $("breadcrumbs");
      if (!root) return;
      var items = ["МО Аналитика", PAGE_TITLES[state.page] || state.page].concat(
        state.drillTrail.map(function (entry) { return entry.label; })
      );
      root.innerHTML = "<ol>" + items.map(function (label, index) {
        var current = index === items.length - 1 ? ' aria-current="page"' : "";
        return "<li" + current + ">" + esc(label) + "</li>";
      }).join("") + "</ol>";
    }
    function pushDrill(label, apply) {
      if (!state.drillSnapshot) state.drillSnapshot = selectionSnapshot();
      state.drillTrail.push({ label: label, apply: apply });
      if (state.drillTrail.length > 14) state.drillTrail = state.drillTrail.slice(-14);
      renderAnalysisRail();
    }
    function clearDrillTrail(restore) {
      if (restore && state.drillSnapshot) restoreSelection(state.drillSnapshot);
      state.drillTrail = [];
      state.drillSnapshot = null;
      renderAnalysisRail();
    }
    function applyDrill(options) {
      options = options || {};
      var action = function () {
        if (options.findingCode !== undefined) state.findingCode = options.findingCode;
        if (options.search !== undefined) state.search = options.search;
        if (options.caseSearchValue !== undefined) $("case-search").value = options.caseSearchValue;
        if (options.selected) {
          Object.keys(options.selected).forEach(function (key) {
            state.selected[key] = options.selected[key];
          });
        }
        if (options.period) {
          state.period = options.period;
          $("period").value = state.period;
          $("date-from-wrap").hidden = state.period !== "custom";
          $("date-to-wrap").hidden = state.period !== "custom";
        }
        if (options.dateFrom !== undefined) { state.dateFrom = options.dateFrom; $("date-from").value = state.dateFrom; }
        if (options.dateTo !== undefined) { state.dateTo = options.dateTo; $("date-to").value = state.dateTo; }
        renderChips();
        switchPage(options.page || state.page);
      };
      pushDrill(options.label || "drill-down", action);
      action();
    }
    function renderChips() {
      var html = [];
      if (state.search) {
        html.push('<span class="chip chip-search">Поиск: ' + esc(state.search) +
          '<button type="button" data-clear-search aria-label="Очистить поиск">×</button></span>');
      }
      Object.keys(state.selected).forEach(function (key) {
        state.selected[key].forEach(function (value) {
          html.push('<span class="chip">' + esc(FILTER_LABELS[key] + ": " + value) +
            '<button type="button" data-remove="' + esc(key) + '" data-value="' + esc(value) + '" aria-label="Удалить фильтр">×</button></span>');
        });
      });
      if (state.findingCode) {
        html.push('<span class="chip">Замечание: ' + esc(state.findingCode) +
          '<button type="button" data-clear-finding aria-label="Удалить фильтр замечания">×</button></span>');
      }
      if (state.rubricCriterion) {
        html.push('<span class="chip">Рубрика МЗ: ' + esc(state.rubricCriterion) +
          '<button type="button" data-clear-rubric aria-label="Удалить фильтр рубрики">×</button></span>');
      }
      $("filter-chips").innerHTML = html.join("");
      $("filter-chips").querySelectorAll("[data-remove]").forEach(function (button) {
        button.addEventListener("click", function () {
          var key = button.getAttribute("data-remove"), value = button.getAttribute("data-value");
          state.selected[key] = state.selected[key].filter(function (x) { return x !== value; });
          var filter = document.querySelector('.filter-pop[data-filter="' + key + '"]');
          if (filter) renderFilter(filter);
          filtersChanged();
        });
      });
      var clearSearch = $("filter-chips").querySelector("[data-clear-search]");
      if (clearSearch) clearSearch.addEventListener("click", clearCaseSearch);
      var clearFinding = $("filter-chips").querySelector("[data-clear-finding]");
      if (clearFinding) clearFinding.addEventListener("click", function () {
        state.findingCode = "";
        filtersChanged();
      });
      var clearRubric = $("filter-chips").querySelector("[data-clear-rubric]");
      if (clearRubric) clearRubric.addEventListener("click", function () {
        state.rubricCriterion = "";
        renderChips();
      });
    }
    function syncUrl(replace) {
      var q = query();
      q.set("page", state.page);
      var path = state.page === "yesterday" ? "/methodist/mo/yesterday" :
        (state.page === "queue" || state.page === "documents" ? "/methodist/mo/cases" : "/methodist/mo");
      var url = path + "?" + q.toString();
      history[replace ? "replaceState" : "pushState"]({ page: state.page }, "", url);
    }
    function readUrl() {
      var q = new URLSearchParams(location.search);
      var pathPage = location.pathname.endsWith("/yesterday") ? "yesterday" :
        (location.pathname.endsWith("/cases") ? "documents" : "overview");
      state.page = PAGE_TITLES[q.get("page")] ? q.get("page") : pathPage;
      state.period = q.get("period") || "month"; state.compare = q.get("compare_period") || "previous";
      state.dateFrom = q.get("date_from") || ""; state.dateTo = q.get("date_to") || "";
      state.search = q.get("q") || "";
      state.findingCode = q.get("finding_codes") || "";
      state.sortBy = q.get("sort_by") || "date";
      state.sortDir = q.get("sort_dir") || "desc";
      Object.keys(state.selected).forEach(function (key) {
        state.selected[key] = (q.get(API_FILTER_KEYS[key] || key) || "").split(",").filter(Boolean);
      });
      $("period").value = state.period; $("compare").value = state.compare;
      $("date-from").value = state.dateFrom; $("date-to").value = state.dateTo;
      $("case-search").value = state.search;
      $("sort-by").value = state.sortBy;
      $("sort-dir").value = state.sortDir;
      $("date-from-wrap").hidden = state.period !== "custom";
      $("date-to-wrap").hidden = state.period !== "custom";
      updateFilterSummary();
    }
    function filtersChanged() {
      if (state.drillTrail.length) {
        state.drillTrail = [];
        state.drillSnapshot = null;
      }
      state.pageNo = 1;
      renderChips(); updateFilterSummary(); syncUrl(true); loadPage(state.page);
      renderAnalysisRail();
    }
    function switchPage(page, push) {
      if (!PAGE_TITLES[page]) page = "overview";
      state.page = page;
      document.querySelectorAll(".page").forEach(function (section) { section.hidden = section.getAttribute("data-page") !== page; });
      document.querySelectorAll(".nav-button").forEach(function (button) {
        if (button.getAttribute("data-page") === page) button.setAttribute("aria-current", "page");
        else button.removeAttribute("aria-current");
      });
      document.title = PAGE_TITLES[page] + " | МО Аналитика";
      if (push !== false) syncUrl(false);
      renderAnalysisRail();
      loadPage(page);
      $("main").focus({ preventScroll: true });
      window.scrollTo(0, 0);
    }
    function renderTrendChart(element, option, config) {
      return MO.moChart(element, option, config);
    }
    async function loadLegacyOverview(suffix) {
      return request("/overview" + suffix, "__root__");
    }
    function renderMonthTrend(data) {
      var items = (data.timeseries || {}).items || [], dates = items.map(function (item) { return item.date; });
      var names = {
        overall:"Итог", documentation:"Оформление", clinical_concordance:"Клиническая согласованность",
        safety:"Безопасность", regulatory:"Регуляторика"
      };
      var series = Object.keys(names).map(function (key) {
        return { name:names[key], type:"line", connectNulls:true, symbolSize:6,
          data:items.map(function (item) { return item[key]; }) };
      });
      series = series.map(function (item) {
        return Object.assign({}, item, {
          smooth: true,
          showSymbol: false,
          lineStyle: { width: 2.4 },
          areaStyle: item.name === "Итог" ? { opacity: 0.08 } : undefined
        });
      });
      series.push({ name:"Объём", type:"bar", yAxisIndex:1, barMaxWidth:18,
        itemStyle:{ borderRadius:[6,6,0,0], opacity:.45 },
        data:items.map(function (item) {
          return { value:item.volume, itemStyle:item.anomaly ? { color:"#be123c", opacity:1 } : null };
        }), markPoint:{ data:items.map(function (item, index) {
          return item.anomaly ? { name:"Аномалия", coord:[index,item.volume], value:"!" } : null;
        }).filter(Boolean) } });
      var chart = MO.moChart($("month-trend-chart"), {
        tooltip:{ trigger:"axis" }, legend:{ type:"scroll" }, grid:{ left:48,right:54,top:58,bottom:68 },
        dataZoom:[{ type:"inside" },{ type:"slider", bottom:10 }],
        xAxis:{ type:"category", name:"Дата", data:dates },
        yAxis:[{ type:"value", name:"Индекс, %", min:0, max:100 },{ type:"value", name:"Записи" }],
        series:series
      }, { label:"Динамика четырёх индексов и объёма за месяц",
        description:"Линии показывают индексы, столбцы объём, красные маркеры аномальные дни.",
        fallback:function (target) { target.innerHTML=items.map(function (item) { return bar(item.date,item.overall); }).join(""); } });
      if (chart) chart.on("click", function (params) {
        var day = items[params.dataIndex] && items[params.dataIndex].date;
        if (day) applyDrill({ label: "День " + day, period: "custom", dateFrom: day, dateTo: day, page: "documents" });
      });
    }
    function renderMonthHeatmap(data) {
      var cells=((data.heatmap || {}).cells || []), rows=Array.from(new Set(cells.map(function (x) { return x.row; }))),
        cols=Array.from(new Set(cells.map(function (x) { return x.col; })));
      if (!cells.length) { $("month-heatmap-chart").innerHTML=unavailableBlock(data.heatmap); return; }
      var chart=MO.moChart($("month-heatmap-chart"), {
        tooltip:{ formatter:function (p) { var x=cells[p.dataIndex]; return esc(x.row)+"<br>"+esc(x.col)+"<br>Оценка: "+x.avg_score+"%<br>n = "+x.n; } },
        grid:{ left:135,right:22,top:18,bottom:70 }, dataZoom:[{ type:"inside" }],
        xAxis:{ type:"category", name:"Глава МКБ", data:cols }, yAxis:{ type:"category", name:"Специальность", data:rows },
        visualMap:{ min:50,max:100,calculable:true,orient:"horizontal",left:"center",bottom:4 },
        series:[{ type:"heatmap", data:cells.map(function (x) { return [cols.indexOf(x.col),rows.indexOf(x.row),x.avg_score,x.n]; }), label:{ show:true,formatter:function (p) { return p.data[3]; } } }]
      }, { label:"Тепловая карта специальностей и глав МКБ", description:"Цвет означает среднюю оценку, подпись число случаев.",
        fallback:function (target) { target.innerHTML=cells.slice(0,12).map(function (x) { return bar(x.row+" / "+x.col,x.avg_score,x.n); }).join(""); } });
      if (chart) chart.on("click", function (params) {
        var cell=cells[params.dataIndex]; applyDrill({ label: "Специальность " + cell.row, selected: { specialties: [cell.row] }, page: "documents" });
      });
    }
    function renderMonthDoctors(data) {
      var section=data.doctor_case_mix || {}, items=(section.items || []).filter(function (x) {
        return x.enough_data && !x.suppressed && x.delta != null;
      }).slice(0,15);
      if (!items.length) { $("month-doctor-chart").innerHTML=unavailableBlock(section,"Нет врачей, прошедших порог n."); }
      else {
        var chart=MO.moChart($("month-doctor-chart"), {
          tooltip:{ formatter:function (p) { var x=items[p.dataIndex], ci=x.delta_ci95 || {}; return esc(x.label)+"<br>Дельта: "+signed(x.delta)+"<br>95% ДИ: "+signed(ci.low)+" ... "+signed(ci.high)+"<br>n = "+x.n; } },
          grid:{ left:145,right:24,top:18,bottom:42 }, xAxis:{ type:"value",name:"Дельта, п.п." },
          yAxis:{ type:"category",data:items.map(function (x) { return x.label; }) },
          series:[{ type:"bar",barMaxWidth:18,itemStyle:{ borderRadius:[0,8,8,0] },
          data:items.map(function (x) { return x.delta; }),markLine:{ symbol:"none",data:[{ xAxis:0 }] } }]
        }, { label:"Рейтинг врачей по case-mix дельте",description:"Показаны врачи с достаточной выборкой и доверительным интервалом.",
          fallback:function (target) { target.innerHTML=items.map(function (x) { return notice(x.label,signed(x.delta)+", n="+x.n,"review"); }).join(""); } });
        if (chart) chart.on("click",function (p) { applyDrill({ label: "Врач " + items[p.dataIndex].label, selected: { doctors: [items[p.dataIndex].label] }, page: "documents" }); });
      }
      $("month-doctor-note").innerHTML='<p class="inline-note">'+esc(section.rule || "")+"</p>";
    }
    function renderMonthPareto(data) {
      var section=data.pareto || {}, items=section.items || [];
      if (!items.length) { $("month-pareto-chart").innerHTML=unavailableBlock(section); return; }
      var chart=MO.moChart($("month-pareto-chart"), {
        tooltip:{ trigger:"axis", formatter:function (params) {
          var x=items[params[0].dataIndex] || {};
          return esc(x.label || x.finding_code || "") + "<br>" + esc(x.cases) + " случаев";
        } }, grid:{ left:48,right:48,top:28,bottom:110 },
        xAxis:{ type:"category",axisLabel:{ rotate:28, interval:0, formatter:function (value) {
          return String(value || "").length > 28 ? String(value).slice(0, 26) + "…" : value;
        } },data:items.map(function (x) { return x.label || x.finding_code; }) },
        yAxis:[{ type:"value",name:"Случаи" },{ type:"value",name:"Накоплено, %",min:0,max:100 }],
        series:[{ type:"bar",name:"Случаи",barMaxWidth:22,itemStyle:{ borderRadius:[6,6,0,0] },data:items.map(function (x) { return x.cases; }) },
          { type:"line",name:"Накопленная доля",smooth:true,showSymbol:false,yAxisIndex:1,data:items.map(function (x) { return x.cumulative_share_pct; }) }]
      }, { label:"Парето замечаний месяца",description:"Клик открывает документы с этим замечанием.",
        fallback:function (target) { target.innerHTML=items.map(function (x) { return bar(x.label || x.finding_code,x.cumulative_share_pct,x.cases); }).join(""); } });
      if (chart) chart.on("click",function (p) {
        var item = items[p.dataIndex] || {};
        navigateFinding(item.finding_code, item.label || item.finding_code);
      });
    }
    function renderMonthFunnel(data) {
      var funnel=data.funnel || {}, stages=[
        ["Источник",funnel.source],["Допущено",funnel.eligible],["Оценено",funnel.evaluated],
        ["С замечаниями",funnel.with_findings],["В работе CRM",funnel.in_crm_work],["Закрыто",funnel.closed]
      ];
      MO.moChart($("month-funnel-chart"), { tooltip:{ trigger:"item" },
        series:[{ type:"funnel",left:"8%",width:"84%",label:{ formatter:"{b}: {c}" },
          itemStyle:{ borderRadius:6, borderColor:"#fff", borderWidth:1 },
          data:stages.map(function (x) { return { name:x[0],value:x[1] || 0 }; }) }] },
      { label:"Воронка месяца",description:"Путь записей от источника до закрытия в CRM.",
        fallback:function (target) { target.innerHTML=stages.map(function (x) { return bar(x[0],funnel.source ? 100*x[1]/funnel.source : 0,x[1]); }).join(""); } });
      var statuses=(data.crm_progress || {}).statuses || {}, keys=Object.keys(statuses);
      if (!keys.length) $("month-crm-chart").innerHTML=unavailableBlock(data.crm_progress);
      else MO.moChart($("month-crm-chart"), { tooltip:{ trigger:"axis" },grid:{ left:105,right:18,top:18,bottom:40 },
        xAxis:{ type:"value",name:"Случаи" },yAxis:{ type:"category",data:keys.map(statusLabel) },
        series:[{ type:"bar",barMaxWidth:18,itemStyle:{ borderRadius:[0,8,8,0] },data:keys.map(function (key) { return statuses[key]; }) }] },
      { label:"Прогресс CRM по статусам",description:"Количество оценённых случаев в каждом рабочем статусе.",
        fallback:function (target) { target.innerHTML=keys.map(function (key) { return notice(statusLabel(key),statuses[key]+" случаев","good"); }).join(""); } });
    }
    function renderOverview(data) {
      if (!data.available) { showError(data.reason || "Данные месяца недоступны."); return; }
      var summary=normalizeSummary(data), k=data.kpi || {}, forecast=data.forecast || {};
      state.data.summary=summary;
      $("month-period-label").textContent=(data.period_label || "MTD")+" с "+data.period.date_from+" по "+data.data_through+
        ". Дней: "+data.days_elapsed+" из "+data.days_in_month+". Europe/Minsk.";
      $("freshness").textContent="Данные по "+data.data_through;
      $("month-kpis").innerHTML=kpi("Записи MTD",k.source_records,"из БД МИС")+
        kpi("Оценено",k.evaluated,score(k.coverage_pct)+" от допущенных")+
        kpi("Итоговая оценка",score(k.avg_score),"deep / по оценённым")+
        kpi("Рубрика МЗ",score((data.rubric_mz || {}).avg_rubric_pct),"shadow · «Как оценивать»")+
        kpi("Требует внимания",k.needs_attention,(k.needs_attention_pct || 0)+"% оценённых")+
        kpi("Критические",k.critical,"P0 случаи")+
        kpi("Прогноз объёма",forecast.projected_source,"к концу месяца");
      $("month-forecast").innerHTML=kpi("Прогноз записей",forecast.projected_source,forecast.method)+
        kpi("Прогноз оценённых",forecast.projected_evaluated,"при текущем темпе")+
        kpi("Прогноз оценки",score(forecast.projected_avg_score),"без изменения среднего");
      var comparisons=data.comparison || {};
      $("month-compare").innerHTML=Object.keys(comparisons).map(function (key) {
        var item=comparisons[key];
        return item.available ? notice(item.label,
          "Записи "+signed(item.deltas.source_records,"")+"; оценка "+signed(item.deltas.avg_score),"good") :
          notice("Сравнение недоступно",item.reason,"review");
      }).join("")+"<p class=\"inline-note\">"+esc((forecast.assumptions || []).join(". "))+"</p>";
      var reconciliation=data.reconciliation || {}, banner=$("month-reconciliation");
      banner.hidden=reconciliation.status === "ok";
      banner.className="banner critical";
      banner.textContent="Расхождение дневных и MTD итогов: источник "+reconciliation.source_delta+
        ", оценено "+reconciliation.evaluated_delta+". Данные не замаскированы.";
      renderMonthTrend(data);renderMonthHeatmap(data);renderMonthDoctors(data);renderMonthPareto(data);renderMonthFunnel(data);
      $("month-reg55").innerHTML=(data.reg55 || {}).available ?
        kpi("Соответствие №55",score(data.reg55.value),"проверенная метрика") : unavailableBlock(data.reg55);
      renderMonthRubricMz(data.rubric_mz);
    }
    function renderMonthRubricMz(rubric) {
      var host = $("month-rubric-mz");
      if (!host) return;
      if (!rubric || !rubric.available) {
        host.innerHTML = unavailableBlock(rubric, "Нет выборки для рубрики МЗ за период.");
        return;
      }
      var top = (rubric.top_fail || []).slice(0, 8).map(function (item) {
        return '<tr tabindex="0" data-rubric-criterion="' + esc(item.id) + '"><td>' + esc(item.title || item.id) + '</td><td>' + esc(item.zero_n) +
          '</td><td>' + esc(item.half_n) + '</td><td><b>' + esc(item.fail_pct) + '%</b></td></tr>';
      }).join("");
      var titles = {};
      (rubric.top_fail || []).forEach(function (item) { titles[item.id] = item.title || item.id; });
      var specialtyRows = (rubric.by_specialty || []).slice(0, 8).map(function (row) {
        var weak = (row.top_criteria || []).map(function (c) {
          return esc(titles[c.id] || c.id) + " (" + esc(c.fail_n) + ")";
        }).join("; ");
        return "<tr><td>" + esc(row.specialty) + "</td><td>" + esc(row.fail_n) +
          "</td><td>" + (weak || " - ") + "</td></tr>";
      }).join("");
      host.innerHTML =
        kpi("Средняя рубрика", score(rubric.avg_rubric_pct), "shadow · sample " + (rubric.sample_n || 0)) +
        kpi("Выборка", rubric.sample_n, (rubric.date_from || "") + " - " + (rubric.date_to || "")) +
        '<div class="table-wrap" style="margin-top:10px"><table class="rubric-table"><thead><tr>' +
        '<th>Критерий</th><th>0</th><th>0.5</th><th>Доля слабостей</th></tr></thead><tbody>' +
        (top || '<tr><td colspan="4" class="empty">Слабых критериев нет.</td></tr>') +
        '</tbody></table></div>' +
        '<p class="card-sub">Клик по критерию открывает очередь разбора с подсветкой этого пункта в карточке случая.</p>' +
        '<h3 style="margin:14px 0 8px;font-size:14px">Слабости по специальностям</h3>' +
        '<div class="table-wrap"><table class="rubric-table"><thead><tr>' +
        '<th>Специальность</th><th>Слабых оценок</th><th>Топ критерии</th></tr></thead><tbody>' +
        (specialtyRows || '<tr><td colspan="3" class="empty">Недостаточно данных по специальностям.</td></tr>') +
        '</tbody></table></div>';
      host.querySelectorAll("[data-rubric-criterion]").forEach(function (row) {
        function openCriterion(event) {
          if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") return;
          event.preventDefault();
          state.rubricCriterion = row.getAttribute("data-rubric-criterion") || "";
          renderChips();
          switchPage("queue");
        }
        row.addEventListener("click", openCriterion);
        row.addEventListener("keydown", openCriterion);
      });
    }
    async function loadOverview() {
      var suffix = "?" + query().toString();
      var q = query();
      var rubricQuery = new URLSearchParams();
      if (q.get("date_from")) rubricQuery.set("date_from", q.get("date_from"));
      if (q.get("date_to")) rubricQuery.set("date_to", q.get("date_to"));
      rubricQuery.set("limit", "120");
      var responses = await Promise.all([
        request("/month-report" + suffix, "__root__"),
        request("/facets" + suffix, "/cases" + suffix),
        request("/rubric-summary?" + rubricQuery.toString())
      ]);
      var response = responses[0], facetsResponse = responses[1], rubricResponse = responses[2];
      if (response.status === 401 || response.status === 403) { setAuth(true); return; }
      if (!response.ok) throw new Error("Не удалось загрузить отчёт месяца.");
      var raw = await response.json();
      if (facetsResponse.ok) {
        var facetPayload = await facetsResponse.json();
        raw.facets = facetPayload.facets || facetPayload;
      }
      if (rubricResponse && rubricResponse.ok) {
        raw.rubric_mz = await rubricResponse.json();
      } else {
        raw.rubric_mz = { available: false, reason: "Сводка рубрики МЗ недоступна" };
      }
      renderOverview(raw);
      buildFacets(normalizeSummary(raw), raw.facets);
    }
    function rowRecord(row) {
      var id = row.case_id || row.visit_id || row.id || "";
      var doctor = row.doctor_fio || row.doctor || "Врач не указан";
      var specialty = row.specialization || row.specialty || "";
      var diagnosis = normalizeDiagnosis(row);
      var total = firstNumeric([row.deep_overall_pct, row.overall_pct, row.l1_overall_pct]);
      var fallbackStatus = total == null ? "unscored" : (total < 60 ? "critical" : total < 75 ? "review" : "good");
      var status = row.crm_status || ((row.crm || {}).status) || row.deep_status || row.status || fallbackStatus;
      return { raw: row, id: id, date: row.date || row.visit_date || "", doctor: doctor, specialty: specialty,
        branch: row.filial || row.branch || "", diagnosis: diagnosis, total: total, status: status,
        kind: row.document_kind_label || row.kz_kind_label || row.kz_kind || "Не указан",
        coverage: firstNumeric([row.coverage_pct, row.coverage, row.deep_coverage_pct]),
        confidence: firstNumeric([row.confidence_pct, row.confidence, row.deep_confidence_pct]) };
    }
    function statusLabel(value) {
      var map = { new:"Новый", assigned:"Назначен", in_review:"На разборе", confirmed_issue:"Подтверждено",
        false_positive:"Отклонено", needs_more_data:"Нужны данные", sent_to_doctor:"Передано врачу",
        resolved:"Решено", closed:"Закрыто", critical:"Критично", review:"Требует внимания",
        poor:"Требует внимания", good:"Хорошо", acceptable:"Приемлемо", needs_review:"Требует внимания",
        case_action:"Решение по случаю", bulk_action:"Групповое изменение", unscored:"Не оценено" };
      return map[value] || value || "Не указан";
    }
    function statusClass(value) { return /critical|confirmed/.test(value) ? "critical" : /good|resolved|closed|acceptable/.test(value) ? "good" : "review"; }
    function documentRow(item) {
      return '<tr tabindex="0" data-case="' + esc(item.id) + '"><td>' + esc(item.date) + '</td><td><b>' + esc(item.doctor) +
        '</b><br><small>' + esc(item.specialty) + '</small></td><td>' + esc(item.branch) + '</td><td>' + esc(item.diagnosis) +
        '</td><td>' + esc(item.kind) + '</td><td><b>' + esc(scoreLabel(item.total, item.raw.score_reason)) + '</b></td><td>' + esc(score(item.coverage)) +
        '</td><td>' + esc(score(item.confidence)) + '</td><td><span class="status ' + statusClass(item.status) + '">' +
        esc(statusLabel(item.status)) + "</span></td></tr>";
    }
    function queueRow(item) {
      var priority = Number(item.raw.p0 || 0) > 0 ? "P0" : Number(item.raw.p1 || 0) > 0 ? "P1" : "Низкий балл";
      var crm = item.raw.crm || {};
      var pdfUrl = item.raw.pdf_url || ("/api/methodist/mo/cases/" + encodeURIComponent(item.id) + "/pdf");
      return '<tr tabindex="0" data-case="' + esc(item.id) + '"><td><input type="checkbox" data-case-select="' + esc(item.id) + '" aria-label="Выбрать случай"></td><td><span class="status ' +
        statusClass(item.status) + '">' + esc(priority) + '</span></td><td>' + esc(item.date) +
        '</td><td>' + esc(item.branch) + '</td><td><b>' + esc(item.doctor) + '</b><br><small>' + esc(item.specialty) +
        '</small></td><td>' + esc(item.diagnosis) + '</td><td>' + esc(scoreLabel(item.total, item.raw.score_reason)) + '</td><td>' +
        esc(item.raw.reason || item.raw.comment || "Требует ручной проверки") + '</td><td>' +
        esc(item.raw.assignee || crm.assignee || "Не назначен") + '</td><td>' + esc(item.raw.due_date || crm.due_date || "Сегодня") +
        '</td><td>' + esc(statusLabel(item.status)) +
        '</td><td class="row-actions"><button class="button secondary compact" type="button" data-open-pdf="' + esc(pdfUrl) + '" data-open-name="mo-' + esc(item.id) + '.pdf">МО в PDF</button></td></tr>';
    }
    function bindCaseRows(container) {
      container.querySelectorAll("[data-case]").forEach(function (row) {
        function open(event) {
          if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") return;
          if (event.target && event.target.closest('input[type="checkbox"], button, a')) return;
          event.preventDefault(); openCase(row.getAttribute("data-case"), row);
        }
        row.addEventListener("click", open); row.addEventListener("keydown", open);
      });
    }
    async function loadCases(queue) {
      var q = query(); q.set("page", state.pageNo); q.set("page_size", 50);
      if (queue) q.set("queue_only", "1");
      var response = await request("/cases?" + q.toString(), "/cases?" + q.toString());
      if (!response.ok) throw new Error("Не удалось загрузить случаи.");
      var data = await response.json();
      var rows = (data.rows || data.cases || data.items || data.worst_visits || []).map(rowRecord);
      var body = queue ? $("queue-rows") : $("document-rows");
      var emptyState = data.empty_state || {};
      body.innerHTML = rows.length ? rows.map(queue ? queueRow : documentRow).join("") :
        '<tr><td colspan="' + (queue ? 12 : 9) + '" class="empty"><b>' +
        esc(emptyState.title || "По выбранным фильтрам случаев нет.") + "</b><div>" +
        esc(emptyState.hint || "Измените фильтры или расширьте период.") + "</div></td></tr>";
      bindCaseRows(body);
      applyColumnVisibility(queue ? "queue" : "documents");
      if (!queue) {
        var total = Number(data.total || rows.length), pages = Math.max(1, Math.ceil(total / 50));
        $("pager").innerHTML = '<span>Страница ' + state.pageNo + " из " + pages + " · всего " + esc(total) +
          '</span><button class="button secondary" id="previous-page"' + (state.pageNo <= 1 ? " disabled" : "") +
          '>Предыдущая</button><button class="button secondary" id="next-page"' +
          (state.pageNo >= pages ? " disabled" : "") + '>Следующая</button>';
        $("previous-page").addEventListener("click", function () {
          if (state.pageNo > 1) { state.pageNo -= 1; loadCases(false); }
        });
        $("next-page").addEventListener("click", function () {
          if (state.pageNo < pages) { state.pageNo += 1; loadCases(false); }
        });
      }
      if (data.facets) buildFacets(state.data.summary || normalizeSummary({}), data.facets);
    }
    async function openCase(id, trigger) {
      if (!id) return;
      state.openCaseId = id;
      state.trigger = trigger;
      $("case-drawer").hidden = false; $("drawer-backdrop").hidden = false;
      document.body.style.overflow = "hidden";
      $("drawer-body").innerHTML = '<div class="skeleton"></div>';
      $("drawer-close").focus();
      var q = query(); q.set("month", q.get("month") || minskDateKey(0).slice(0,7)); q.set("visit_id", id);
      try {
        var response = await request("/cases/" + encodeURIComponent(id), "/case-detail?" + q.toString());
        if (!response.ok) throw new Error("Случай не найден.");
        renderCase(await response.json());
      } catch (e) { $("drawer-body").innerHTML = '<div class="banner">' + esc(e.message) + "</div>"; }
    }
    function renderCase(data) {
      var record = data.record || data.case || data;
      var item = rowRecord(record);
      var axes = data.axes || {};
      var findings = data.findings || record.findings || [];
      var crm = data.crm || record.crm || {};
      var events = data.events || [];
      var coverageInfo = deriveCoverage(data, record, axes);
      var confidenceInfo = deriveConfidence(data, record, axes);
      var sourceDocument = data.document || {};
      var crmStatus = crm.status || "new";
      var statusOptions = [
        ["new","Новый"],["assigned","Назначен"],["in_review","На разборе"],
        ["confirmed_issue","Подтверждено"],["false_positive","Отклонено"],
        ["needs_more_data","Нужны данные"],["sent_to_doctor","Передано врачу"],
        ["resolved","Решено"],["closed","Закрыто"]
      ].map(function (option) {
        return '<option value="' + option[0] + '"' + (option[0] === crmStatus ? " selected" : "") + ">" + option[1] + "</option>";
      }).join("");
      $("drawer-title").textContent = "Разбор случая";
      $("drawer-subtitle").textContent = [item.date, item.doctor, item.specialty, item.branch].filter(Boolean).join(" · ");
      var rubric = data.rubric_mz || {};
      $("drawer-body").innerHTML =
        '<div class="drawer-grid">' + kpi("Итоговая оценка", score(data.deep_overall_pct != null ? data.deep_overall_pct : item.total), "по доступным данным") +
        kpi("Рубрика МЗ", score(rubric.rubric_pct), rubric.primary ? "методика «Как оценивать»" : "shadow · «Как оценивать»") +
        kpi("Статус", statusLabel(data.deep_status || item.status), "рабочий статус") +
        kpi("Полнота проверки", score(coverageInfo.value), coverageInfo.estimated ? "оценка по доступным полям" : "доступность исходных данных") +
        kpi("Надёжность", score(confidenceInfo.value), confidenceInfo.estimated ? "оценка по доступным полям" : "устойчивость результата") + '</div>' +
        '<div class="detail-block"><h3>Оси оценки</h3>' + ["documentation","clinical_concordance","safety","regulatory"].map(function (key) {
          var labels = { documentation:"Оформление", clinical_concordance:"Согласованность", safety:"Безопасность", regulatory:"Регуляторика" };
          return bar(labels[key], axes[key] == null ? record["axis_" + key] : axes[key]);
        }).join("") + '</div>' +
        renderRubricMz(rubric) +
        renderClinicalDocument(sourceDocument, findings) +
        '<div class="detail-block"><h3>Выявленные замечания</h3>' + (findings.length ? findings.map(function (finding) {
          var title = finding.title_ru || finding.title || finding.code || "Замечание";
          var decision = (crm.finding_decisions || {})[finding.code] || "unreviewed";
          var linked = finding.linked_fields || [];
          var shadowBadge = (finding.is_shadow || finding.shadow) ?
            '<span class="status review finding-shadow-badge">shadow</span> ' : "";
          var linkHint = finding.link_hint_ru ?
            '<p class="finding-link-hint">' + esc(finding.link_hint_ru) +
            (linked.length ? ' · поля: ' + linked.map(function (field) {
              return '<button type="button" class="linkish" data-focus-clinical="' + esc(field) + '">' +
                esc(clinicalFieldLabel(field)) + '</button>';
            }).join(", ") : "") + '</p>' : "";
          return notice(finding.severity || "Проверить", shadowBadge + (title || finding.detail_ru || finding.detail || "Требуется ручная проверка"),
            finding.severity === "P0" ? "critical" : "review") +
            ((finding.detail_ru || finding.detail) ? '<p>' + esc(finding.detail_ru || finding.detail) + '</p>' : "") +
            linkHint +
            (finding.evidence ? '<blockquote>«' + esc(finding.evidence) + '»</blockquote>' : "") +
            (finding.evidence_span ? '<p class="card-sub">Поле ' + esc(finding.evidence_span.field) +
              ', символы ' + esc(finding.evidence_span.start) + '-' + esc(finding.evidence_span.end) + '</p>' : "") +
            (finding.source_ref ? '<details><summary>Источник</summary><p>' + esc(finding.source_ref) + "</p></details>" : "") +
            (finding.code ? '<label class="filter"><span>Решение по замечанию</span><select class="control" data-finding-code="' +
              esc(finding.code) + '"><option value="unreviewed"' + (decision === "unreviewed" ? " selected" : "") +
              '>Не проверено</option><option value="confirmed"' + (decision === "confirmed" ? " selected" : "") +
              '>Подтверждено</option><option value="false_positive"' + (decision === "false_positive" ? " selected" : "") +
              '>Отклонено</option><option value="needs_more_data"' + (decision === "needs_more_data" ? " selected" : "") +
              '>Нужны данные</option></select></label>' : "");
        }).join("") : '<p>Критических замечаний не найдено.</p>') + '</div>' +
        '<div class="detail-block"><h3>Решение методиста</h3><label class="filter"><span>Статус</span><select class="control" id="drawer-status">' + statusOptions +
        '</select></label><label class="filter"><span>Ответственный</span><input class="control" id="drawer-assignee" maxlength="120" value="' +
        esc(crm.assignee || "") + '"></label><label class="filter"><span>Срок</span><input class="control" id="drawer-due" type="date" value="' +
        esc(crm.due_date || "") + '"></label><label class="filter"><span>Метки через запятую</span><input class="control" id="drawer-tags" maxlength="500" value="' +
        esc((crm.tags || []).join(", ")) + '"></label><label class="filter" style="margin-top:10px"><span>Комментарий</span><input class="control" id="drawer-comment" maxlength="2000"></label>' +
        '<p><button class="button" id="drawer-save" type="button">Сохранить решение</button> ' +
        '<a class="button secondary" href="/doctor/review?source=mo&amp;case=' + encodeURIComponent(item.id) + '">Анализ документа</a> ' +
        '<button class="button secondary" type="button" data-open-pdf="/api/methodist/mo/cases/' + encodeURIComponent(item.id) + '/pdf" data-open-name="mo-' + encodeURIComponent(item.id) + '.pdf">МО в PDF</button></p></div>' +
        '<div class="detail-block"><h3>История решений</h3>' + (events.length ? events.map(function (event) {
          return notice(new Date(event.created_at).toLocaleString("ru-RU"), statusLabel(event.event_type) + " · " + (event.actor || "методист"), "good");
        }).join("") : '<p class="empty">Решений пока нет.</p>') + '</div>';
      $("drawer-save").addEventListener("click", saveCaseDecision);
      $("drawer-body").querySelectorAll("[data-focus-clinical]").forEach(function (button) {
        button.addEventListener("click", function () {
          var field = button.getAttribute("data-focus-clinical");
          var target = $("drawer-body").querySelector('[data-clinical-field="' + field + '"]');
          if (!target) return;
          $("drawer-body").querySelectorAll(".clinical-field--linked").forEach(function (node) {
            node.classList.remove("clinical-field--focus");
          });
          target.classList.add("clinical-field--focus");
          target.scrollIntoView({ block: "nearest", behavior: "smooth" });
        });
      });
      var focusRow = document.getElementById("rubric-focus-row");
      if (focusRow) focusRow.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
    function clinicalFieldLabel(key) {
      return ({
        complaints: "Жалобы", anamnesis_doctor: "Анамнез", anamnesis_auto: "Анамнез (авто)",
        objective_status: "Объективный статус", exam_data: "Данные обследований",
        clinical_diagnosis: "Клинический диагноз", mis_diagnos: "Диагноз МИС",
        mkb_code_main: "МКБ", exam_recommendations: "Рекомендации по обследованию",
        treatment_recommendations: "Рекомендации по лечению"
      })[key] || key;
    }
    function renderRubricMz(rubric) {
      if (!rubric || !rubric.ok) {
        return '<div class="detail-block"><h3>Рубрика МЗ («Как оценивать»)</h3>' +
          '<p class="empty">Shadow-оценка по методике МЗ пока недоступна для этого случая.</p></div>';
      }
      var groupLabels = {
        documentation: "Документация", clinical: "Клиника",
        dynamics: "Динамика", regulatory: "Регламент"
      };
      var rows = (rubric.criteria || []).map(function (item) {
        var label = item.score_label == null ? "n/a" : String(item.score_label);
        var tone = label === "1" ? "good" : (label === "0.5" ? "review" : (label === "0" ? "critical" : "muted"));
        var focus = state.rubricCriterion && state.rubricCriterion === item.id ? " rubric-row--focus" : "";
        return '<tr class="rubric-row rubric-row--' + tone + focus + '"' +
          (focus ? ' id="rubric-focus-row"' : "") + '>' +
          '<td><span class="rubric-score rubric-score--' + tone + '">' + esc(label) + '</span></td>' +
          '<td><div class="rubric-title">' + esc(item.title || item.id || "") + '</div>' +
            '<div class="card-sub">' + esc(groupLabels[item.group] || item.group || "") +
            (item.scored_by_55 ? " · №55" : " · №127") + '</div></td>' +
          '<td><div>' + esc(item.reason || "") + '</div>' +
            (item.how_to_evaluate ? '<div class="card-sub">Как оценивать: ' + esc(item.how_to_evaluate) + '</div>' : "") +
          '</td></tr>';
      }).join("");
      var focusNote = state.rubricCriterion ?
        '<p class="inline-note">Фокус очереди: критерий «' + esc(state.rubricCriterion) + '». Серверный фильтр по рубрике - после записи в warehouse.</p>' : "";
      return '<div class="detail-block"><h3>Рубрика МЗ («Как оценивать»)</h3>' + focusNote +
        '<p class="card-sub">Shadow · ' + esc(rubric.scorer_version || "mz-rubric") +
        ' · оценено ' + esc(rubric.scored_n != null ? rubric.scored_n : " - ") +
        ', n/a ' + esc(rubric.na_n != null ? rubric.na_n : " - ") +
        ' · итог ' + esc(score(rubric.rubric_pct)) +
        (rubric.prior_available ? ' · prior ' + esc(rubric.prior_visit_date || "") : ' · prior n/a') +
        '</p>' +
        '<div class="table-wrap"><table class="rubric-table"><thead><tr>' +
        '<th>Оценка</th><th>Параметр</th><th>Пояснение</th></tr></thead><tbody>' +
        (rows || '<tr><td colspan="3" class="empty">Критерии не рассчитаны.</td></tr>') +
        '</tbody></table></div></div>';
    }
    function renderClinicalDocument(documentData, findings) {
      var clinical = documentData.clinical || {};
      var linkedSet = {};
      (findings || []).forEach(function (finding) {
        (finding.linked_fields || []).forEach(function (field) { linkedSet[field] = true; });
      });
      var fields = [
        ["complaints", "Жалобы"], ["anamnesis_doctor", "Анамнез"],
        ["anamnesis_auto", "Анамнез (авто)"], ["objective_status", "Объективный статус"],
        ["exam_data", "Данные обследований"], ["manipulations", "Манипуляции"],
        ["clinical_diagnosis", "Клинический диагноз"], ["mis_diagnos", "Диагноз МИС"],
        ["exam_recommendations", "Рекомендации по обследованию"],
        ["treatment_recommendations", "Рекомендации по лечению"]
      ];
      var content = fields.filter(function (field) { return clinical[field[0]]; }).map(function (field) {
        var linked = linkedSet[field[0]] ? " clinical-field--linked" : "";
        return '<section class="clinical-field' + linked + '" data-clinical-field="' + esc(field[0]) + '"><h4>' +
          esc(field[1]) + (linkedSet[field[0]] ? ' <span class="clinical-link-mark">↔ замечание</span>' : "") +
          '</h4><p>' + esc(clinical[field[0]]) + '</p></section>';
      }).join("");
      var reason = documentData.score_reason ? '<p class="inline-note">' + esc(documentData.score_reason) + '</p>' : "";
      var sourceLabel = documentData.source_format === "secure_csv" ? "защищённый дневной срез" :
        (documentData.source_format === "parquet" ? "дневной parquet" : "источник не определён");
      return '<div class="detail-block"><h3>Исходное МО</h3>' + reason +
        (content || '<div class="empty"><b>Клинический текст недоступен</b><div>Проверены опубликованные источники за дату визита. Откройте визит в МИС.</div></div>') +
        '<p class="card-sub">Источник: ' + esc(sourceLabel) + '</p>' +
        '<p class="card-sub">Жалобы и анамнез → статус → диагноз → МКБ → обследования → лечение → наблюдение</p></div>';
    }
    async function postCaseChanges(caseIds, changes, comment) {
      var response = await request("/cases/bulk-action", "/cases/bulk-action", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ case_ids: caseIds, changes: changes, comment: comment || "" })
      });
      if (!response.ok) throw new Error("Не удалось сохранить изменения.");
      return response.json();
    }
    async function saveCaseDecision() {
      try {
        var findingDecisions = {};
        document.querySelectorAll("[data-finding-code]").forEach(function (select) {
          findingDecisions[select.getAttribute("data-finding-code")] = select.value;
        });
        await postCaseChanges([state.openCaseId], {
          status: $("drawer-status").value,
          assignee: $("drawer-assignee").value.trim(),
          due_date: $("drawer-due").value,
          tags: $("drawer-tags").value.split(",").map(function (tag) { return tag.trim(); }).filter(Boolean),
          finding_decisions: findingDecisions
        }, $("drawer-comment").value);
        $("announcer").textContent = "Решение сохранено";
        closeDrawer(); loadPage(state.page);
      } catch (e) { showError(e.message); }
    }
    function selectedCaseIds() {
      return Array.from(document.querySelectorAll('#queue-rows input[data-case-select]:checked')).map(function (input) {
        return input.getAttribute("data-case-select");
      });
    }
    async function bulkChange(changes) {
      var ids = selectedCaseIds();
      if (!ids.length) { showError("Выберите хотя бы один случай."); return; }
      try {
        await postCaseChanges(ids, changes, "");
        $("announcer").textContent = "Обновлено случаев: " + ids.length;
        loadCases(true);
      } catch (e) { showError(e.message); }
    }
    function closeDrawer() {
      $("case-drawer").hidden = true; $("drawer-backdrop").hidden = true; document.body.style.overflow = "";
      if (state.trigger) state.trigger.focus();
    }
    function unavailableBlock(section, fallback) {
      return '<div class="empty"><b>Показатель недоступен</b><div>' +
        esc((section || {}).reason || fallback || "Данных для расчёта нет.") + "</div></div>";
    }
    function signed(value, suffix) {
      if (value == null) return "нет сравнения";
      return (Number(value) > 0 ? "+" : "") + Number(value).toFixed(1) + (suffix || " п.п.");
    }
    function renderYesterdayCompleteness(data) {
      var completeness = data.data_completeness || {};
      var reasonLabels = {
        scoring_coverage: "оценены не все допущенные записи",
        scoring_errors: "ошибки оценки",
        llm_queue_pending: "очередь LLM"
      };
      function reasonText(codes) {
        return (codes || []).map(function (code) {
          return reasonLabels[code] || code;
        }).join(", ");
      }
      if (!completeness.available) {
        $("yesterday-completeness").innerHTML = unavailableBlock(completeness);
      } else {
        var expected = completeness.expected_rows || {};
        var statusLine = completeness.partial
          ? "день помечен как неполный" + (completeness.partial_reasons && completeness.partial_reasons.length
            ? " (" + reasonText(completeness.partial_reasons) + ")"
            : "")
          : "день завершён";
        if (!completeness.partial && completeness.advisory_reasons && completeness.advisory_reasons.length) {
          statusLine += " · замечание: " + reasonText(completeness.advisory_reasons);
          if (completeness.llm_queue_pending) {
            statusLine += " (" + completeness.llm_queue_pending + ")";
          }
        }
        $("yesterday-completeness").innerHTML =
          kpi("Получено", completeness.actual_rows, "строк из источника") +
          kpi("Ожидалось", expected.available ? expected.value : "Нет базы", expected.available ? expected.samples + " сопоставимых дней" : expected.reason) +
          notice("Лаг", completeness.lag_days + " дн. · ревизия " + (completeness.revision == null ? "не указана" : completeness.revision) +
            " · " + statusLine,
            completeness.partial ? "critical" : (completeness.advisory_reasons && completeness.advisory_reasons.length ? "review" : "good")) +
          (completeness.flags || []).map(function (flag) {
            return notice(flag.level === "blocking" ? "Блокирующий флаг" : "Предупреждение",
              flag.message || flag.code, flag.level === "blocking" ? "critical" : "review");
          }).join("");
      }
      var funnel = data.funnel || {};
      $("yesterday-funnel").innerHTML = funnel.available ?
        kpi("Источник", funnel.source, "все типы документов") +
        kpi("Допущено", funnel.eligible, "медосмотры и консультации") +
        kpi("Оценено", funnel.evaluated, "оценка рассчитана") +
        kpi("Исключено", funnel.excluded, "не входит в оценку") : unavailableBlock(funnel);
      $("yesterday-kind-rows").innerHTML = (funnel.document_kinds || []).length ?
        funnel.document_kinds.map(function (item) {
          if (item.suppressed) return "<tr><td>" + esc(item.label) + '</td><td colspan="4">Скрыто: группа ' + esc(item.n_bucket) + "</td></tr>";
          return "<tr><td>" + esc(item.label) + "</td><td>" + esc(item.source) + "</td><td>" + esc(item.eligible) +
            "</td><td>" + esc(item.evaluated) + "</td><td>" + esc(item.excluded) + "</td></tr>";
        }).join("") : '<tr><td colspan="5" class="empty">Разбивка по типам документов недоступна.</td></tr>';
    }
    function renderYesterdayIndices(data) {
      var items = ((data.indices || {}).items || []);
      $("yesterday-index-cards").innerHTML = items.map(function (item) {
        return kpi(item.label, item.available ? score(item.value) : "Нет данных",
          "к предыдущему дню: " + signed(item.delta_previous_day),
          item.delta_weekday_mean == null ? "" : signed(item.delta_weekday_mean) + " к среднему дня недели");
      }).join("") || unavailableBlock(data.indices);
      var available = items.filter(function (item) { return item.available; });
      var chart = MO.moChart($("yesterday-index-chart"), {
        tooltip: { trigger: "axis" },
        legend: { data: ["За день", "Предыдущий день", "Среднее дня недели"] },
        grid: { left: 48, right: 18, top: 46, bottom: 52 },
        xAxis: { type: "category", name: "Индекс", data: available.map(function (item) { return item.label; }) },
        yAxis: { type: "value", name: "Оценка, %", min: 0, max: 100 },
        series: [
          { name: "За день", type: "bar", barMaxWidth: 28, itemStyle: { borderRadius: [6,6,0,0] }, data: available.map(function (item) { return item.value; }) },
          { name: "Предыдущий день", type: "bar", barMaxWidth: 28, itemStyle: { borderRadius: [6,6,0,0] }, data: available.map(function (item) { return item.previous_day; }) },
          { name: "Среднее дня недели", type: "line", smooth: true, symbolSize: 9, data: available.map(function (item) { return item.weekday_mean_8w; }) }
        ]
      }, {
        label: "Сравнение четырёх индексов за вчера",
        description: "Для каждого индекса показано значение дня, предыдущего дня и среднее того же дня недели.",
        fallback: function (target) {
          target.innerHTML = available.map(function (item) { return bar(item.label, item.value); }).join("");
        }
      });
      if (chart) chart.on("click", function () { applyDrill({ label: "Индексы за вчера", page: "documents" }); });
    }
    function navigateFinding(code, sourceLabel) {
      applyDrill({
        label: sourceLabel || ("Замечание " + (code || "")),
        findingCode: code || "",
        search: "",
        caseSearchValue: "",
        page: "documents"
      });
    }
    function navigateYesterdayFinding(code, label, day) {
      applyDrill({
        label: label || ("Замечание " + (code || "")),
        findingCode: code || "",
        search: "",
        caseSearchValue: "",
        period: day ? "custom" : state.period,
        dateFrom: day || state.dateFrom,
        dateTo: day || state.dateTo,
        page: "documents"
      });
    }
    function renderYesterdayFindings(data) {
      var items = ((data.top_findings || {}).items || []).slice(0, 12);
      var day = (data.top_findings || {}).day || data.day || data.data_through || "";
      if (!items.length) {
        $("yesterday-findings-chart").innerHTML = unavailableBlock(data.top_findings);
        $("yesterday-findings-list").innerHTML = "";
        return;
      }
      var total = items.reduce(function (sum, item) { return sum + Number(item.cases || 0); }, 0), running = 0;
      var cumulative = items.map(function (item) { running += Number(item.cases || 0); return total ? Math.round(1000 * running / total) / 10 : 0; });
      var chartLabels = items.map(function (item) { return item.label || item.finding_code; });
      var chart = MO.moChart($("yesterday-findings-chart"), {
        tooltip: { trigger: "axis", formatter: function (params) {
          var item = items[params[0].dataIndex] || {};
          return esc(item.label || item.finding_code || "") + "<br>" + esc(item.severity || "") +
            " · " + esc(item.cases) + " случаев";
        } },
        grid: { left: 48, right: 48, top: 30, bottom: 110 },
        xAxis: { type: "category", name: "Замечание", axisLabel: { rotate: 28, interval: 0, formatter: function (value) {
          return String(value || "").length > 28 ? String(value).slice(0, 26) + "…" : value;
        } }, data: chartLabels },
        yAxis: [{ type: "value", name: "Случаи" }, { type: "value", name: "Накоплено, %", min: 0, max: 100 }],
        series: [
          { name: "Случаи", type: "bar", data: items.map(function (item) {
            return { value: item.cases, itemStyle: { decal: { symbol: item.severity === "P0" ? "rect" : "line" } } };
          }) },
          { name: "Накопленная доля", type: "line", yAxisIndex: 1, data: cumulative }
        ]
      }, {
        label: "Парето замечаний за вчера",
        description: "Клик открывает список МО с этим замечанием за день.",
        fallback: function (target) {
          target.innerHTML = items.map(function (item) { return bar(item.label || item.finding_code, Math.min(100, item.cases), item.cases); }).join("");
        }
      });
      if (chart) chart.on("click", function (params) {
        var item = items[params.dataIndex] || {};
        navigateYesterdayFinding(item.finding_code, item.label, day);
      });
      $("yesterday-findings-list").innerHTML = items.map(function (item) {
        var samples = (item.sample_cases || []).slice(0, 5).map(function (sample) {
          return '<button class="finding-case-link" type="button" data-open-case="' + esc(sample.case_id) + '">' +
            esc(sample.doctor || sample.case_id) +
            (sample.specialty ? ' <small>' + esc(sample.specialty) + '</small>' : '') +
            '</button>';
        }).join("");
        return '<div class="finding-card">' +
          '<button class="finding-link" type="button" data-yesterday-finding="' + esc(item.finding_code) +
          '" data-yesterday-label="' + esc(item.label || item.finding_code) +
          '" data-yesterday-day="' + esc(day) + '">' +
          '<span class="status ' + (item.severity === "P0" || item.severity === "P1" ? "critical" : "review") + '">' +
          esc(item.severity) + '</span> <b>' + esc(item.label || item.finding_code) + '</b>' +
          '<span class="finding-meta">' + esc(item.cases) + ' случаев · открыть список МО</span></button>' +
          (samples ? '<div class="finding-cases">' + samples + '</div>' : '') +
          '</div>';
      }).join("");
    }
    function renderYesterdayDoctors(data) {
      var section = data.doctor_outliers || {}, items = section.items || [];
      if (!items.length) {
        $("yesterday-doctor-chart").innerHTML = unavailableBlock(section);
        $("yesterday-doctor-note").innerHTML = "";
        return;
      }
      var chart = MO.moChart($("yesterday-doctor-chart"), {
        tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, formatter: function (params) {
          var item = items[params[0].dataIndex], ci = item.delta_ci95 || {};
          return esc(item.label) + "<br>Дельта: " + signed(item.delta) + "<br>95% ДИ: " +
            (ci.low == null ? "недоступен" : signed(ci.low) + " ... " + signed(ci.high)) + "<br>n = " + item.n;
        } },
        grid: { left: 145, right: 28, top: 20, bottom: 42 },
        xAxis: { type: "value", name: "Дельта к ожидаемой, п.п." },
        yAxis: { type: "category", name: "Врач", data: items.map(function (item) { return item.label; }) },
        series: [{ name: "Дельта", type: "bar", data: items.map(function (item) { return item.delta; }),
          markLine: { symbol: "none", data: [{ xAxis: -10, name: "Порог -10" }] } }]
      }, {
        label: "Врачи с оценкой ниже ожидаемой",
        description: "Ранжирование по дельте к ожидаемой оценке своей специальности.",
        fallback: function (target) {
          target.innerHTML = items.map(function (item) { return notice(item.label, signed(item.delta) + ", n=" + item.n, "review"); }).join("");
        }
      });
      if (chart) chart.on("click", function (params) {
        var doctor = items[params.dataIndex].label;
        state.findingCode = "";
        applyDrill({ label: "Врач " + doctor, selected: { doctors: [doctor] }, page: "documents" });
      });
      $("yesterday-doctor-note").innerHTML = '<p class="inline-note">' + esc(section.rule) + "</p>";
    }
    function renderYesterdayActions(data) {
      var section = data.action_cases || {}, items = section.items || [];
      $("yesterday-action-rows").innerHTML = items.length ? items.map(function (item) {
        var pdfUrl = item.pdf_url || ("/api/methodist/mo/cases/" + encodeURIComponent(item.case_id) + "/pdf");
        return '<tr data-case="' + esc(item.case_id) + '"><td><span class="status ' +
          (item.severity === "P0" ? "critical" : "review") + '">' + esc(item.severity) +
          "</span></td><td><b>" + esc(item.doctor) + "</b><br><small>" + esc(item.specialty) +
          "</small></td><td>" + esc(item.branch) + "</td><td>" + esc(item.diagnosis) +
          "</td><td><b>" + esc(item.finding_title || item.finding_code) + "</b>" +
          (item.is_shadow ? ' <span class="status review finding-shadow-badge">shadow</span>' : "") +
          "<br><small>" + esc(item.reason) +
          (item.overall_pct != null ? " · оценка " + Math.round(Number(item.overall_pct)) + "%" : "") +
          '</small></td><td class="row-actions"><button class="button secondary compact" type="button" data-take-case="' +
          esc(item.case_id) + '"' + (item.crm_status === "in_review" ? " disabled" : "") + ">" +
          (item.crm_status === "in_review" ? "Уже в работе" : "Взять в работу") +
          '</button> <button class="button secondary compact" type="button" data-open-pdf="' + esc(pdfUrl) + '" data-open-name="mo-' + esc(item.case_id) + '.pdf">МО в PDF</button></td></tr>';
      }).join("") : '<tr><td colspan="6">' + unavailableBlock(section, "P0/P1 случаев нет.") + "</td></tr>";
      bindCaseRows($("yesterday-action-rows"));
    }
    function renderYesterdayFlow(data, dimension) {
      var section = data.flow_changes || {}, dimensions = section.dimensions || {};
      var items = (dimensions[dimension] || []).filter(function (item) { return item.available; }).slice(0, 12);
      if (!items.length) {
        $("yesterday-flow-chart").innerHTML = unavailableBlock(section, "Нет публикуемых групп в этом разрезе.");
        $("yesterday-flow-note").innerHTML = "";
        return;
      }
      var chart = MO.moChart($("yesterday-flow-chart"), {
        tooltip: { trigger: "axis" },
        legend: { data: ["За день", "Предыдущий день"] },
        grid: { left: 48, right: 18, top: 45, bottom: 95 },
        xAxis: { type: "category", name: "Группа", axisLabel: { rotate: 30 }, data: items.map(function (item) { return item.key; }) },
        yAxis: { type: "value", name: "Доля потока, %" },
        series: [
          { name: "За день", type: "bar", data: items.map(function (item) { return item.share_pct; }) },
          { name: "Предыдущий день", type: "bar", data: items.map(function (item) { return item.previous_share_pct; }) }
        ]
      }, {
        label: "Состав потока и изменение против предыдущего дня",
        description: "Сравниваются доли групп в текущем и предыдущем полном дне.",
        fallback: function (target) {
          target.innerHTML = items.map(function (item) { return bar(item.key, item.share_pct); }).join("");
        }
      });
      if (chart) chart.on("click", function (params) {
        var stateKey = dimension === "specialty" ? "specialties" : dimension === "branch" ? "branches" : "document_types";
        state.findingCode = "";
        applyDrill({ label: "Поток " + items[params.dataIndex].key, selected: (function(){ var obj={}; obj[stateKey]=[items[params.dataIndex].key]; return obj; })(), page: "documents" });
      });
      $("yesterday-flow-note").innerHTML = items.slice(0, 4).map(function (item) {
        return notice(item.key, "Доля " + item.share_pct + "%, изменение " + signed(item.share_delta_pp), Math.abs(item.share_delta_pp || 0) >= 5 ? "review" : "good");
      }).join("");
    }
    function renderYesterdaySourceQuality(data) {
      var section = data.source_quality || {}, items = section.items || [];
      $("yesterday-source-quality").innerHTML = items.length ? items.map(function (item) {
        if (!item.available) return notice(item.label, item.reason, "review");
        var inverse = item.key === "date_mismatch_pct";
        return bar(item.label, inverse ? Math.max(0, 100 - Number(item.value)) : item.value,
          Number(item.value).toFixed(1) + "%" + (inverse ? " расхождений" : ""));
      }).join("") : unavailableBlock(section);
    }
    async function takeYesterdayCase(caseId, button) {
      button.disabled = true;
      try {
        await postCaseChanges([caseId], { status: "in_review" }, "Взято в работу из отчёта за вчера");
        button.textContent = "Уже в работе";
        showToast("Случай " + caseId + " взят в работу");
      } catch (error) {
        button.disabled = false;
        showError(error.message);
      }
    }
    function renderYesterday(data) {
      renderYesterdayCompleteness(data);
      renderYesterdayIndices(data);
      renderYesterdayFindings(data);
      renderYesterdayDoctors(data);
      renderYesterdayActions(data);
      renderYesterdayFlow(data, $("yesterday-flow-dimension").value);
      renderYesterdaySourceQuality(data);
    }
    async function loadYesterday() {
      var day = (state.period === "custom" && state.dateFrom) ? state.dateFrom : minskDateKey(-1);
      $("yesterday-date").textContent = "Итоги за " + new Date(day + "T12:00:00").toLocaleDateString("ru-RU", { dateStyle:"long" }) + ".";
      var response = await request("/daily-report?date=" + encodeURIComponent(day), "__root__");
      if (response.status === 401 || response.status === 403) { setAuth(true); return; }
      if (!response.ok) throw new Error("Отчёт за " + day + " пока недоступен.");
      var data = await response.json();
      state.data.daily = data;
      $("partial-banner").hidden = !(data.partial || data.quality_status === "blocked");
      var completeness = data.data_completeness || {};
      var banner = $("partial-banner");
      if (!banner.hidden) {
        var reasons = (completeness.partial_reasons || []).join(", ");
        banner.textContent = data.quality_status === "blocked"
          ? "Данные заблокированы проверкой качества. Итоговый отчёт не принимается."
          : ("Данные неполные" + (reasons ? " (" + reasons + ")" : "") +
            ". Итог за день доделывается; цифры ниже могут быть неполными.");
      } else if (completeness.advisory_reasons && completeness.advisory_reasons.length) {
        banner.hidden = false;
        banner.textContent = "День принят с замечанием: " +
          completeness.advisory_reasons.join(", ") +
          (completeness.llm_queue_pending ? " (" + completeness.llm_queue_pending + ")" : "") +
          ". Очередь LLM не блокирует итог при достаточном покрытии.";
      }
      renderYesterday(data);
    }
    function renderEntityPages(summary) {
      $("diagnosis-findings").innerHTML = (summary.findings || []).slice(0,8).map(function (x) {
        return notice(x.severity || "Проверить", x.title || x.label || "Требуется ручная проверка", x.severity === "P0" ? "critical" : "review");
      }).join("") || '<div class="empty">Замечаний по выбранному срезу нет.</div>';
      $("quality-kpis").innerHTML = kpi("Свежесть", summary.generated ? "Актуально" : "Нет отметки", summary.generated) +
        kpi("Записей", summary.n, "получено") + kpi("Оценено", summary.evaluated, "после проверки") + kpi("Пропуски", Math.max(0, summary.n-summary.evaluated), "не допущено");
      $("quality-chart").innerHTML = bar("Обработано", summary.n ? summary.evaluated / summary.n * 100 : 0) +
        bar("Полнота проверки", summary.coverage || 0) + bar("Надёжность", summary.confidence || 0);
      $("quality-warnings").innerHTML = notice(summary.generated ? "Загрузка завершена" : "Нет времени обновления",
        summary.generated || "Проверьте источник данных", summary.generated ? "good" : "review");
    }
    async function dimensionData(name) {
      var response = await request("/dimensions/" + name + "?" + query().toString(), "/dimensions/" + name);
      if (!response.ok) throw new Error("Не удалось загрузить интерактивный разрез.");
      return response.json();
    }
    async function loadDoctorsDimension() {
      var data = await dimensionData("doctors"), items = data.items || [];
      $("doctor-rows").innerHTML = items.length ? items.map(function (x) {
        var ci = x.delta_ci95 || {};
        return '<tr data-doctor-key="' + esc(x.key) + '"><td><button class="link-button" data-open-doctor="' +
          esc(x.key) + '"><b>' + esc(x.label) + "</b></button></td><td>" + esc(x.specialty) +
          "</td><td>" + esc(x.n == null ? x.n_bucket : x.n) + "</td><td>" +
          esc(x.enough_data ? signed(x.delta) : "Мало данных") + "</td><td>" +
          esc(x.enough_data ? signed(ci.low) + " - " + signed(ci.high) : "Недоступно") +
          "</td><td>" + esc(x.p0_cases == null ? "Скрыто" : x.p0_cases) + "</td></tr>";
      }).join("") : '<tr><td colspan="6" class="empty">Нет данных по врачам.</td></tr>';
      var plotted = items.filter(function (x) { return x.enough_data && !x.suppressed && x.delta != null; });
      var chart = MO.moChart($("doctor-scatter-chart"), {
        tooltip:{ formatter:function (p) { var x=plotted[p.dataIndex], ci=x.delta_ci95 || {};
          return esc(x.label)+"<br>Объём: "+x.n+"<br>Дельта: "+signed(x.delta)+
            "<br>95% ДИ: "+signed(ci.low)+" - "+signed(ci.high)+"<br>P0: "+(x.p0_cases || 0); } },
        toolbox:{ feature:{ brush:{ type:["rect","clear"] }, dataZoom:{}, saveAsImage:{} } },
        brush:{ toolbox:["rect","clear"], xAxisIndex:"all", yAxisIndex:"all" },
        grid:{ left:58,right:30,top:55,bottom:55 },
        xAxis:{ type:"value", name:"Число записей" },
        yAxis:{ type:"value", name:"Дельта к ожидаемой, п.п.", axisLine:{ onZero:true } },
        series:[{ type:"scatter", data:plotted.map(function (x) {
          return { value:[x.n,x.delta,Math.max(8,Math.min(42,8+(x.p0_cases || 0)*4))], doctor:x };
        }), symbolSize:function (value) { return value[2]; } }]
      }, { label:"Врачи: объём и дельта к ожидаемой оценке",
        description:"Каждая точка - врач с выборкой не меньше двадцати записей. Размер означает число P0." });
      function openDoctor(key) { state.cabinetDoctorKey=key; switchPage("doctor-cabinet"); }
      $("doctor-rows").querySelectorAll("[data-open-doctor]").forEach(function (button) {
        button.addEventListener("click",function () {
          var key = button.getAttribute("data-open-doctor");
          if (!key) return;
          pushDrill("Кабинет врача", function () { openDoctor(key); });
          openDoctor(key);
        });
      });
      if (chart) {
        chart.on("click",function (params) {
          if (!plotted[params.dataIndex]) return;
          var key = plotted[params.dataIndex].key;
          pushDrill("Кабинет врача", function () { openDoctor(key); });
          openDoctor(key);
        });
        chart.on("brushSelected",function (params) {
          var selected=[], batches=(params.batch && params.batch[0] && params.batch[0].selected) || [];
          batches.forEach(function (batch) { (batch.dataIndex || []).forEach(function (index) {
            if (plotted[index] && selected.indexOf(plotted[index]) < 0) selected.push(plotted[index]);
          }); });
          $("doctor-selection-flow").innerHTML=selected.length ?
            "<p><b>Выбрано врачей: "+selected.length+"</b></p><p>"+selected.map(function (x) { return esc(x.label); }).join(", ")+
            '</p><button class="button" id="open-selected-doctors">Открыть их случаи</button>' :
            "Выделите точки рамкой. Действие не выполняется автоматически.";
          var action=$("open-selected-doctors");
          if (action) action.addEventListener("click",function () {
            applyDrill({ label: "Группа врачей", selected: { doctors: selected.map(function (x) { return x.label; }) }, page: "documents" });
          });
        });
      }
    }
    async function loadSpecialtiesDimension() {
      var data=await dimensionData("specialties"), items=data.items || [];
      var chart=MO.moChart($("specialty-boxplot-chart"),{
        tooltip:{ trigger:"item",formatter:function (p) { var x=items[p.dataIndex];
          return esc(x.label)+"<br>Мин / Q1 / медиана / Q3 / макс<br>"+x.boxplot.join(" / ")+"<br>n = "+x.n; } },
        grid:{ left:175,right:30,top:25,bottom:45 }, xAxis:{ type:"value",name:"Оценка, %",min:0,max:100 },
        yAxis:{ type:"category",data:items.map(function (x) { return x.label; }) },
        series:[{ type:"boxplot",data:items.map(function (x) { return x.boxplot; }) }]
      },{ label:"Распределение оценок по специальностям",description:"Коробчатые диаграммы показывают квартили и диапазон." });
      $("specialty-attention").innerHTML=items.length ? '<p class="card-sub">Показано групп: '+items.length+". Малые группы скрыты.</p>" : '<div class="empty">Нет групп выше порога публикации.</div>';
      if (chart) chart.on("click",function (params) {
        var item=items[params.dataIndex]; if (!item) return;
        applyDrill({ label: "Специальность " + item.key, selected: { specialties: [item.key] }, page: "doctors" });
      });
    }
    async function loadDiagnosesDimension() {
      var data=await dimensionData("diagnoses"), items=data.items || [];
      var chart=MO.moChart($("icd-treemap-chart"),{
        tooltip:{ formatter:function (p) { return esc(p.name)+"<br>Объём: "+p.value+
          (p.data.score == null ? "" : "<br>Средняя оценка: "+p.data.score+"%"); } },
        visualMap:{ min:50,max:100,dimension:2,calculable:true,orient:"horizontal",left:"center",bottom:4 },
        series:[{ type:"treemap",roam:true,nodeClick:"zoomToNode",data:items.map(function (chapter) {
          return Object.assign({},chapter,{ children:(chapter.children || []).map(function (child) {
            return Object.assign({},child,{ value:[child.value,child.value,child.score] });
          }) });
        }), levels:[{}, { itemStyle:{ borderWidth:3,gapWidth:3 } }, { itemStyle:{ borderWidth:1,gapWidth:1 } }] }]
      },{ label:"Дерево глав и кодов МКБ",description:"Площадь означает объём, цвет среднюю оценку." });
      if (chart) chart.on("click",function (params) {
        var drill=params.data && params.data.drilldown;
        if (drill && drill.level === "diagnosis") applyDrill({ label: "МКБ " + drill.id, search: drill.id, caseSearchValue: drill.id, page: "documents" });
      });
    }
    async function loadSafetyDimension() {
      var data=await dimensionData("safety"), items=data.items || [], levels=["P0","P1","P2","P3"];
      $("safety-kpis").innerHTML=levels.map(function (level) {
        return kpi(level,items.reduce(function (sum,row) { return sum+(row[level] || 0); },0),"случаев с замечанием");
      }).join("");
      var incidents=data.incidents || [];
      MO.moChart($("safety-severity-chart"),{
        tooltip:{ trigger:"axis" },legend:{ data:levels },grid:{ left:50,right:25,top:50,bottom:55 },
        xAxis:{ type:"category",data:items.map(function (x) { return x.date; }) },yAxis:{ type:"value",name:"Случаи" },
        series:levels.map(function (level) { return { name:level,type:"bar",stack:"severity",
          data:items.map(function (x) { return x[level] || 0; }),
          markPoint:level==="P0" ? { data:incidents.map(function (x) {
            return { name:x.finding_code,coord:[x.date,0],value:"!" };
          }) } : undefined }; })
      },{ label:"Замечания по приоритету по дням",description:"Столбцы сложены по приоритету, маркеры обозначают P0." });
      $("safety-list").innerHTML=incidents.slice(0,30).map(function (x) {
        return notice("P0 · "+x.finding_code,x.date+" · источник: "+(x.source_ref || "не указан"),"critical");
      }).join("") || '<div class="empty">Инцидентов P0 в выбранном периоде нет.</div>';
    }
    async function loadDoctorCabinet() {
      if (!state.cabinetDoctorKey) {
        $("doctor-cabinet-unavailable").hidden=false; $("doctor-cabinet-content").hidden=true; return;
      }
      var response=await request("/doctor-cabinet?doctor_key="+encodeURIComponent(state.cabinetDoctorKey),"/doctor-cabinet");
      if (!response.ok) {
        $("doctor-cabinet-unavailable").hidden=false; $("doctor-cabinet-content").hidden=true;
        $("doctor-cabinet-unavailable").textContent="Кабинет недоступен для текущей роли или доверенная идентификация не настроена.";
        return;
      }
      var data=await response.json(), findings=data.findings || [], byCase={};
      findings.forEach(function (finding) { (byCase[finding.mis_id]||(byCase[finding.mis_id]=[])).push(finding); });
      $("doctor-cabinet-unavailable").hidden=true; $("doctor-cabinet-content").hidden=false;
      $("doctor-cabinet-kpis").innerHTML=kpi("Врач",data.doctor.doctor_fio,data.doctor.specialty)+
        kpi("Записей",(data.cases || []).length,"доступный период")+
        kpi("Замечаний",findings.length,"с цитатами и источниками")+
        kpi("Споров",(data.dispute_stats || {}).total || 0,"передано методисту");
      $("doctor-cabinet-records").innerHTML=(data.cases || []).map(function (item) {
        var caseFindings=byCase[item.mis_id] || [];
        var caseId = item.case_id || item.visit_id || item.mis_id;
        var title = item.title || [item.visit_date, item.diagnosis_code || "Без кода МКБ", item.document_kind_label].filter(Boolean).join(" · ");
        var pdfUrl = item.pdf_url || ("/api/methodist/mo/cases/" + encodeURIComponent(caseId) + "/pdf");
        return '<div class="case-card" data-case="' + esc(caseId) + '"><b>'+esc(title)+
          "</b><p>Оценка: "+esc(scoreLabel(item.overall_pct, item.score_reason))+
          '</p><div class="row-actions"><button class="button secondary compact" type="button" data-open-pdf="'+esc(pdfUrl)+'" data-open-name="mo-'+esc(caseId)+'.pdf">МО в PDF</button></div>'+caseFindings.map(function (finding) {
            return '<div class="finding"><b>'+esc(finding.severity+" · "+(finding.title_ru || finding.finding_code))+
              "</b><p>Источник: "+esc(finding.source_ref || "не указан")+
              '</p><button class="button secondary compact" data-dispute-case="'+esc(item.visit_id || item.mis_id)+
              '" data-dispute-finding="'+esc(finding.finding_code)+'">Оспорить</button></div>';
          }).join("")+"</div>";
      }).join("") || '<div class="empty">Оцениваемых записей нет.' +
        (data.hidden_unscored ? ' Скрыто непрофильных: ' + data.hidden_unscored + '.' : '') + '</div>';
      bindCaseRows($("doctor-cabinet-records"));
      $("doctor-cabinet-actions").innerHTML=(data.what_to_fix || []).map(function (code) {
        return notice(code,"Откройте запись и сверите замечание с указанным источником.","review");
      }).join("") || '<div class="empty">Активных рекомендаций нет.</div>';
      $("doctor-template-pairs").innerHTML=(data.template_pairs || []).map(function (pair) {
        return notice("Сходство "+Math.round(pair.similarity*100)+"%",
          "Случаи "+pair.case_id_a+" и "+pair.case_id_b+" · "+pair.algorithm+" · порог "+pair.threshold,"review");
      }).join("") || '<div class="empty">Шаблонных пар не найдено.</div>';
      $("doctor-cabinet-records").querySelectorAll("[data-dispute-case]").forEach(function (button) {
        button.addEventListener("click",async function () {
          var reason=prompt("Причина оспаривания для методиста"); if (!reason) return;
          var result=await request("/doctor-cabinet/disputes?doctor_key="+encodeURIComponent(state.cabinetDoctorKey),
            "/doctor-cabinet/disputes",{ method:"POST",headers:{ "Content-Type":"application/json" },
              body:JSON.stringify({ case_id:button.getAttribute("data-dispute-case"),
                finding_code:button.getAttribute("data-dispute-finding"),reason:reason }) });
          if (!result.ok) { showError("Не удалось передать спор методисту."); return; }
          showToast("Спор передан методисту"); loadDoctorCabinet();
        });
      });
    }
    async function loadAccessLog() {
      var response=await request("/access-log","/access-log");
      if (response.status===403) { $("access-log-content").innerHTML='<div class="empty">Журнал доступен только администратору.</div>'; return; }
      if (!response.ok) throw new Error("Не удалось загрузить журнал доступа.");
      var data=await response.json();
      $("access-log-content").innerHTML=(data.items || []).map(function (item) {
        return notice(item.action,item.created_at+" · "+item.actor+" · роль "+item.role+
          (item.doctor_key ? " · врач "+item.doctor_key : ""),"good");
      }).join("") || '<div class="empty">Событий доступа пока нет.</div>';
    }
    async function ensureSummary() {
      if (!state.data.summary) await loadOverview();
      renderEntityPages(state.data.summary);
    }
    async function loadDataQuality() {
      var responses = await Promise.all([
        request("/data-quality?" + query().toString(), "/cases?" + query().toString()),
        request("/health", "/freshness")
      ]);
      var response = responses[0], healthResponse = responses[1];
      if (!response.ok) throw new Error("Не удалось загрузить качество данных.");
      var data = await response.json();
      var health = healthResponse.ok ? await healthResponse.json() : {};
      $("quality-kpis").innerHTML =
        kpi("Записей", data.n == null ? data.n_bucket : data.n, "в выбранном срезе") +
        kpi("Распознано", score(data.parse_rate), "структура документа") +
        kpi("Расхождение дат", score(data.date_mismatch_rate), "меньше - лучше") +
        kpi("Без кода МКБ", data.missing_mkb, "требует проверки");
      $("quality-chart").innerHTML =
        bar("Распознано", data.parse_rate || 0) +
        bar("Врач заполнен", data.n ? 100 * (data.n - data.missing_doctor) / data.n : 0) +
        bar("Специальность заполнена", data.n ? 100 * (data.n - data.missing_specialty) / data.n : 0) +
        bar("Филиал заполнен", data.n ? 100 * (data.n - data.missing_branch) / data.n : 0);
      $("quality-warnings").innerHTML =
        notice("Дубли", (data.duplicate_case_ids || 0) + " повторов идентификатора", data.duplicate_case_ids ? "critical" : "good") +
        notice("Неопределённый тип", (data.unknown_document_kind || 0) + " записей", data.unknown_document_kind ? "review" : "good") +
        ((data.empty_state || {}).reason_code && (data.empty_state || {}).reason_code !== "ok"
          ? notice("Пояснение", (data.empty_state || {}).title + ". " + ((data.empty_state || {}).hint || ""), "review")
          : "");
      var labels = {
        warehouse: "Витрина", freshness: "Свежесть", reports: "Ежедневные отчёты",
        case_document_source: "Источник МО", pipeline: "Pipeline"
      };
      var statusLabels = { ready:"Готово", fresh:"Актуально", success:"Успешно", degraded:"Требует внимания", stale:"Устарело", critical:"Критично", missing:"Нет источника", unknown:"Нет данных" };
      $("health-components").innerHTML = Object.keys(health.components || {}).map(function (key) {
        var component = health.components[key] || {}, value = component.status || "unknown";
        var detail = key === "case_document_source" ? ((component.formats || []).join(", ") || "клинический источник не опубликован") :
          (key === "freshness" ? (component.data_through || "дата не определена") :
          (key === "reports" ? (component.missing_days || 0) + " пропущенных отчётов" : "проверено сервером"));
        var tone = /ready|fresh|success/.test(value) ? "good" : (/critical|missing|stale/.test(value) ? "critical" : "review");
        return notice(labels[key] || key, (statusLabels[value] || value) + " · " + detail, tone);
      }).join("") || '<div class="empty">Состояние компонентов недоступно.</div>';
    }

    async function loadCapabilities() {
      try {
        var response = await request("/capabilities", "/meta");
        if (!response.ok) return;
        var capabilities = await response.json();
        state.data.capabilities = capabilities;
        document.querySelectorAll(".nav-button[data-page]").forEach(function (button) {
          var page = button.getAttribute("data-page").replace(/-/g, "_");
          if (Object.prototype.hasOwnProperty.call(capabilities.pages || {}, page)) {
            button.closest("li").hidden = capabilities.pages[page] === false;
          }
        });
      } catch (error) {}
    }
    async function loadReports() {
      var responses = await Promise.all([
        request("/reports", "/dynamics"),
        request("/freshness?" + query().toString(), "/dynamics"),
        request("/month-report?" + query().toString(), "__root__")
      ]);
      var response = responses[0], freshnessResponse = responses[1], monthResponse = responses[2];
      if (!response.ok) throw new Error("Не удалось загрузить список отчётов.");
      var data = await response.json(), items = data.items || [];
      var freshness = freshnessResponse.ok ? await freshnessResponse.json() : (data.freshness || {});
      var month = monthResponse && monthResponse.ok ? await monthResponse.json() : {};
      var kpis = $("report-kpis");
      if (kpis) {
        kpis.innerHTML =
          kpi("Свежесть", freshness.status === "fresh" ? "Актуально" : (freshness.status || "Нет данных"),
            "лаг " + (freshness.lag_days == null ? "н/д" : freshness.lag_days + " дн.")) +
          kpi("Данные до", freshness.data_through || "н/д", "Europe/Minsk") +
          kpi("Отчётов в списке", items.length, "ежедневные готовые срезы") +
          kpi("Оценено за месяц", ((month.kpi || {}).evaluated != null ? month.kpi.evaluated : "н/д"),
            score((month.kpi || {}).avg_score));
      }
      if (!items.length) {
        $("report-list").innerHTML = '<div class="empty">Готовых отчётов пока нет. Дождитесь утреннего приёма данных.</div>';
        return;
      }
      $("report-list").innerHTML = items.map(function (item) {
        var day = item.date || item.month || "";
        var tone = item.quality_status === "blocked" || item.quality_status === "failed" ? "critical" :
          (item.quality_status === "partial" || item.quality_status === "warehouse_only" ? "review" : "good");
        var meta = [];
        if (item.source_rows != null) meta.push("записей " + item.source_rows);
        if (item.evaluated != null) meta.push("оценено " + item.evaluated);
        if (item.avg_score != null) meta.push("средняя " + Math.round(Number(item.avg_score)) + "%");
        if (item.critical != null) meta.push("крит. " + item.critical);
        if (item.needs_attention != null) meta.push("внимание " + item.needs_attention);
        if (item.empty_reason) meta.push(item.empty_reason);
        return '<button class="report-card" type="button" data-report-date="' + esc(day) + '">' +
          '<strong>' + esc(day) + '</strong>' +
          '<span class="status ' + tone + '">' + esc(statusLabel(item.quality_status || "готов")) + '</span>' +
          '<span>Ревизия ' + esc(item.revision || (item.has_report_file === false ? "витрина" : 1)) +
          (item.generated_at ? ' · ' + esc(String(item.generated_at).replace("T", " ").replace("Z", " UTC")) : "") +
          '</span>' +
          (meta.length ? '<span class="report-meta">' + esc(meta.join(" · ")) + '</span>' : '') +
          '</button>';
      }).join("");
      $("report-list").querySelectorAll("[data-report-date]").forEach(function (button) {
        button.addEventListener("click", function () {
          var day = button.getAttribute("data-report-date");
          if (!day) return;
          applyDrill({ label: "Отчёт " + day, period: "custom", dateFrom: day, dateTo: day, page: "yesterday" });
        });
      });
    }
    function currentReportDay() {
      if (state.period === "custom" && state.dateFrom) return state.dateFrom;
      return minskDateKey(-1);
    }
    async function loadScoringMethod() {
      await ensureSummary();
      var responses = await Promise.all([
        request("/scoring-method", "/scoring-info"),
        request("/llm-costs?" + query(), "")
      ]);
      var response = responses[0];
      if (!response.ok) throw new Error("Не удалось загрузить методику оценки.");
      var data = await response.json();
      var axes = (data.axes || []).map(function (axis) {
        var weight = axis.weight == null ? "" : " · вес " + Math.round(Number(axis.weight) * 100) + "%";
        return notice((axis.label || axis.key) + weight, axis.desc || "", "good");
      }).join("");
      var gates = (data.risk_gate || []).map(function (rule) {
        return "<li>" + esc(rule) + "</li>";
      }).join("");
      $("scoring-method").innerHTML =
        '<p><b>Методика:</b> ' + esc(data.scorer_version || "") + ' · веса ' +
        esc(data.weights_version || "") + '</p><p><b>Итог:</b> ' + esc(data.overall_rule || "") +
        '</p><div class="grid"><div class="span-6">' +
        axes + '</div><div class="span-6"><h3>Правила клинического риска</h3><ol>' + gates +
        '</ol><p class="card-sub">Пороговые значения: хорошо ' + esc((data.thresholds || {}).good) +
        ', приемлемо ' + esc((data.thresholds || {}).acceptable) + '.</p></div></div>';
      var costResponse = responses[1];
      if (!costResponse.ok) {
        $("llm-costs").innerHTML = '<div class="empty">Расходы пока недоступны.</div>';
        return;
      }
      var costs = await costResponse.json();
      var rows = (costs.items || []).map(function (item) {
        return "<tr><td>" + esc(item.usage_date) + "</td><td>" + esc(item.tier) +
          "</td><td>" + esc(item.model) + "</td><td>" + fmt(item.calls) +
          "</td><td>" + fmt(item.prompt_tokens) + " / " + fmt(item.completion_tokens) +
          "</td><td>$" + Number(item.cost_usd || 0).toFixed(4) + "</td><td>" +
          fmt(item.avg_latency_ms) + " мс</td></tr>";
      }).join("");
      $("llm-costs").innerHTML =
        '<div class="kpi-row"><div class="kpi"><span>Вызовы</span><b>' + fmt(costs.calls) +
        '</b></div><div class="kpi"><span>Случаи</span><b>' + fmt(costs.cases) +
        '</b></div><div class="kpi"><span>Итого</span><b>$' +
        Number(costs.total_usd || 0).toFixed(4) + '</b></div><div class="kpi"><span>На случай</span><b>$' +
        Number(costs.cost_per_case_usd || 0).toFixed(4) +
        '</b></div></div><div class="table-wrap"><table><thead><tr><th>Дата</th><th>Тир</th><th>Модель</th><th>Вызовы</th><th>Токены вход / выход</th><th>Стоимость</th><th>Задержка</th></tr></thead><tbody>' +
        (rows || '<tr><td colspan="7">LLM-вызовов за период не было.</td></tr>') +
        '</tbody></table></div>';
    }
    async function loadPage(page) {
      $("global-error").hidden = true;
      try {
        if (page === "overview") await loadOverview();
        else if (page === "yesterday") await loadYesterday();
        else if (page === "queue") await loadCases(true);
        else if (page === "documents") await loadCases(false);
        else if (page === "doctors") await loadDoctorsDimension();
        else if (page === "specialties") await loadSpecialtiesDimension();
        else if (page === "diagnoses") await loadDiagnosesDimension();
        else if (page === "safety") await loadSafetyDimension();
        else if (page === "doctor-cabinet") await loadDoctorCabinet();
        else if (page === "access-log") await loadAccessLog();
        else if (page === "data-quality") await loadDataQuality();
        else if (page === "reports") await loadReports();
        else if (page === "settings") await loadScoringMethod();
        else await ensureSummary();
      } catch (e) { showError(e.message || String(e)); }
    }
    function savedViews() {
      if (state.data.views) return state.data.views;
      try { return JSON.parse(localStorage.getItem(VIEWS_KEY) || "[]"); } catch (e) { return []; }
    }
    async function refreshSavedViews() {
      try {
        var response = await request("/saved-views", "/saved-views");
        if (response.ok) state.data.views = (await response.json()).items || [];
      } catch (e) {}
      renderSavedViews();
    }
    function renderSavedViews() {
      var views = savedViews();
      $("saved-view").innerHTML = '<option value="">Текущий срез</option>' + views.map(function (v, i) {
        return '<option value="' + esc(v.view_id || ("local:" + i)) + '">' + esc(v.name) + "</option>";
      }).join("");
      $("view-manager").innerHTML = views.length ? views.map(function (v, i) {
        var key = v.view_id || ("local:" + i);
        return '<div class="view-row"><button class="button secondary" data-load-view="' +
          esc(key) + '">' + esc(v.name) + '</button><button class="button secondary danger" data-delete-view="' + esc(key) + '">Удалить</button></div>';
      }).join("") : '<p class="empty">Сохранённых представлений пока нет.</p>';
    }
    async function saveView() {
      var name = prompt("Название представления");
      if (!name) return;
      var filters = {};
      query().forEach(function (value, key) { filters[key] = value; });
      try {
        var response = await request("/saved-views", "/saved-views", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: name, scope: "private", filters: filters, config: { page: state.page } })
        });
        if (!response.ok) throw new Error("Не удалось сохранить представление");
        await refreshSavedViews();
      } catch (e) {
        var views = savedViews();
        views.push({ name: name, url: location.search });
        localStorage.setItem(VIEWS_KEY, JSON.stringify(views));
        state.data.views = null; renderSavedViews();
      }
      $("announcer").textContent = "Представление сохранено";
    }
    function viewByKey(key) {
      if (String(key).indexOf("local:") === 0) return savedViews()[Number(String(key).slice(6))];
      return savedViews().find(function (view) { return view.view_id === key; });
    }
    function filtersToSearchParams(filters) {
      var params = new URLSearchParams();
      Object.keys(filters || {}).forEach(function (key) {
        var value = filters[key];
        if (value == null || value === "") return;
        params.set(key, Array.isArray(value) ? value.join(",") : String(value));
      });
      return params;
    }
    function loadView(key) {
      var view = viewByKey(key); if (!view) return;
      var suffix = view.url;
      if (!suffix) {
        var params = filtersToSearchParams(view.filters || {});
        params.set("page", ((view.config || {}).page) || "overview");
        suffix = "?" + params.toString();
      }
      history.pushState({}, "", location.pathname + suffix); readUrl(); renderChips(); switchPage(state.page, false);
    }
    async function deleteView(key) {
      var view = viewByKey(key); if (!view) return;
      if (view.view_id) {
        var response = await request("/saved-views/" + encodeURIComponent(view.view_id), "/saved-views/" + encodeURIComponent(view.view_id), { method: "DELETE" });
        if (!response.ok) { showError("Не удалось удалить представление."); return; }
        await refreshSavedViews();
      } else {
        var views = savedViews();
        var index = Number(String(key).slice(6));
        views.splice(index, 1);
        localStorage.setItem(VIEWS_KEY, JSON.stringify(views));
        renderSavedViews();
      }
      $("announcer").textContent = "Представление удалено";
    }
    function commandItems() {
      var items = Object.keys(PAGE_TITLES).map(function (page) {
        return { label: "Перейти: " + PAGE_TITLES[page], action: function () { switchPage(page); } };
      });
      items.push(
        { label: "Период: вчера", action: function () { state.period = "yesterday"; $("period").value = state.period; filtersChanged(); } },
        { label: "Период: последние 7 дней", action: function () { state.period = "7d"; $("period").value = state.period; filtersChanged(); } },
        { label: "Период: текущий месяц", action: function () { state.period = "month"; $("period").value = state.period; filtersChanged(); } },
        { label: "Показать критические случаи", action: function () {
          state.selected.statuses = ["Критично"]; renderChips(); switchPage("queue");
        } }
      );
      savedViews().forEach(function (view, index) {
        items.push({ label: "Представление: " + view.name, action: function () { loadView(view.view_id || ("local:" + index)); } });
      });
      return items;
    }
    function renderCommands(term) {
      term = (term || "").trim().toLowerCase();
      var items = commandItems().filter(function (item) { return item.label.toLowerCase().indexOf(term) >= 0; });
      $("command-results").innerHTML = items.map(function (item, index) {
        return '<button class="command-item" type="button" role="option" data-command="' + index +
          '" aria-selected="' + (index === 0 ? "true" : "false") + '">' + esc(item.label) + "</button>";
      }).join("") || '<p class="empty">Команды не найдены.</p>';
      $("command-results").querySelectorAll("[data-command]").forEach(function (button) {
        button.addEventListener("click", function () {
          var item = items[Number(button.getAttribute("data-command"))];
          closeCommandPalette();
          if (item) item.action();
        });
      });
    }
    function openCommandPalette(trigger) {
      state.commandTrigger = trigger || document.activeElement;
      $("command-palette").hidden = false;
      $("command-backdrop").hidden = false;
      $("command-search").value = "";
      renderCommands("");
      $("command-search").focus();
    }
    function closeCommandPalette() {
      $("command-palette").hidden = true;
      $("command-backdrop").hidden = true;
      if (state.commandTrigger && typeof state.commandTrigger.focus === "function") state.commandTrigger.focus();
    }
    function moveOption(container, direction) {
      var options = Array.from(container.querySelectorAll('[role="option"]'));
      if (!options.length) return;
      var current = options.indexOf(document.activeElement);
      if (current < 0) current = direction > 0 ? -1 : 0;
      current = (current + direction + options.length) % options.length;
      options.forEach(function (option, index) { option.setAttribute("aria-selected", index === current ? "true" : "false"); });
      options[current].focus();
    }
    var COLUMN_MAP = {
      documents: ["Дата", "Врач / специальность", "Филиал", "Диагноз", "Тип документа", "Итог", "Полнота", "Надёжность", "Статус"],
      queue: ["Выбор", "Приоритет", "Дата", "Филиал", "Врач / специальность", "Диагноз", "Итог", "Причина", "Ответственный", "Срок", "Статус", "МО"]
    };
    function ensureColumnState() {
      if (!state.columnVisible.documents.length) state.columnVisible.documents = COLUMN_MAP.documents.map(function () { return true; });
      if (!state.columnVisible.queue.length) state.columnVisible.queue = COLUMN_MAP.queue.map(function () { return true; });
    }
    function applyColumnVisibility(key) {
      ensureColumnState();
      var table = key === "queue" ? document.querySelector("#page-queue table") : document.querySelector("#page-documents table");
      if (!table) return;
      var visible = state.columnVisible[key] || [];
      table.querySelectorAll("tr").forEach(function (row) {
        Array.from(row.children).forEach(function (cell, idx) {
          cell.style.display = visible[idx] === false ? "none" : "";
        });
      });
    }
    function renderColumnsManager() {
      ensureColumnState();
      function block(key, targetId) {
        var host = $(targetId);
        host.innerHTML = '<h3>' + (key === "queue" ? "Очередь" : "Все случаи") + '</h3>' +
          '<div class="filter-options">' + COLUMN_MAP[key].map(function (label, idx) {
            return '<label class="filter-option"><input type="checkbox" data-col-key="' + key + '" data-col-index="' + idx + '"' +
              (state.columnVisible[key][idx] === false ? '' : ' checked') + '><span>' + esc(label) + '</span></label>';
          }).join('') + '</div>';
      }
      block("documents", "columns-manager-doc");
      block("queue", "columns-manager-queue");
      $("columns-manager").querySelectorAll("[data-col-key]").forEach(function (input) {
        input.addEventListener("change", function () {
          var key = input.getAttribute("data-col-key");
          var idx = Number(input.getAttribute("data-col-index"));
          state.columnVisible[key][idx] = !!input.checked;
          applyColumnVisibility(key);
        });
      });
    }
    async function openSelectedQueuePdfs() {
      var ids = selectedCases();
      if (!ids.length) { showToast("Выберите случаи в очереди"); return; }
      if (ids.length > 12 && !confirm("Открыть " + ids.length + " PDF?")) return;
      for (var i = 0; i < ids.length; i++) {
        var caseId = ids[i];
        try {
          await openPdfWithToken('/api/methodist/mo/cases/' + encodeURIComponent(caseId) + '/pdf', { preferredName: 'mo-' + caseId + '.pdf' });
        } catch (error) {
          showError('Не удалось открыть PDF для ' + caseId + ': ' + error.message);
          break;
        }
        if (i < ids.length - 1) await new Promise(function (resolve) { window.setTimeout(resolve, 180); });
      }
    }
    function renderSearchSuggestions(term) {
      var source = [];
      ["doctors", "specialties", "branches"].forEach(function (key) {
        (state.facets[key] || []).forEach(function (item) {
          source.push({ label: item.label, type: FILTER_LABELS[key], value: item.value });
        });
      });
      term = term.trim().toLowerCase();
      var matches = term ? source.filter(function (item) { return item.label.toLowerCase().indexOf(term) >= 0; }).slice(0, 7) : [];
      var box = $("search-suggestions");
      box.innerHTML = matches.map(function (item, index) {
        return '<button class="suggestion" type="button" role="option" data-suggestion="' + esc(item.value) +
          '" aria-selected="' + (index === 0 ? "true" : "false") + '"><b>' + esc(item.label) +
          "</b><small> " + esc(item.type) + "</small></button>";
      }).join("");
      box.hidden = !matches.length;
      $("case-search").setAttribute("aria-expanded", matches.length ? "true" : "false");
      box.querySelectorAll("[data-suggestion]").forEach(function (button) {
        button.addEventListener("click", function () {
          $("case-search").value = button.getAttribute("data-suggestion");
          box.hidden = true;
          $("case-search").setAttribute("aria-expanded", "false");
          updateFilterSummary();
          $("case-search-submit").focus();
        });
      });
    }
    function bind() {
      document.querySelectorAll(".nav-button").forEach(function (button) {
        button.addEventListener("click", function () { switchPage(button.getAttribute("data-page")); });
      });
      document.querySelectorAll("[data-go]").forEach(function (button) {
        button.addEventListener("click", function () { switchPage(button.getAttribute("data-go")); });
      });
      document.querySelectorAll('[data-action="export"]').forEach(function (button) {
        button.addEventListener("click", function () {
          exportCurrent("cases").catch(function (error) { showError(error.message); });
        });
      });
      document.querySelectorAll('[data-action="export-aggregates"]').forEach(function (button) {
        button.addEventListener("click", function () {
          exportCurrent("aggregates").catch(function (error) { showError(error.message); });
        });
      });
      document.querySelectorAll('[data-action="download"]').forEach(function (button) {
        button.addEventListener("click", function () {
          if (state.data.daily) downloadJson(state.data.daily, "mo-daily-" + (state.data.daily.date || "latest") + ".json");
        });
      });
      $("columns-button").addEventListener("click", function () {
        state.columnsPanelOpen = !state.columnsPanelOpen;
        $("columns-manager").hidden = !state.columnsPanelOpen;
        this.setAttribute("aria-pressed", state.columnsPanelOpen ? "true" : "false");
        this.textContent = state.columnsPanelOpen ? "Скрыть настройку колонок" : "Колонки таблиц";
        if (state.columnsPanelOpen) renderColumnsManager();
      });
      $("period").addEventListener("change", function () {
        state.period = this.value;
        $("date-from-wrap").hidden = state.period !== "custom";
        $("date-to-wrap").hidden = state.period !== "custom";
        filtersChanged();
      });
      $("date-from").addEventListener("change", function () { state.dateFrom = this.value; filtersChanged(); });
      $("date-to").addEventListener("change", function () { state.dateTo = this.value; filtersChanged(); });
      $("compare").addEventListener("change", function () { state.compare = this.value; filtersChanged(); });
      $("case-search-form").addEventListener("submit", function (event) {
        event.preventDefault();
        state.search = $("case-search").value.trim();
        $("search-suggestions").hidden = true;
        $("case-search").setAttribute("aria-expanded", "false");
        showToast(state.search ? "Поиск применён: " + state.search : "Поиск очищен");
        filtersChanged();
      });
      $("case-search-clear").addEventListener("click", clearCaseSearch);
      $("sort-by").addEventListener("change", function () { state.sortBy = this.value; filtersChanged(); });
      $("sort-dir").addEventListener("change", function () { state.sortDir = this.value; filtersChanged(); });
      document.querySelectorAll("[data-quick-period]").forEach(function (button) {
        button.addEventListener("click", function () {
          state.period = button.getAttribute("data-quick-period") || "month";
          $("period").value = state.period;
          $("date-from-wrap").hidden = state.period !== "custom";
          $("date-to-wrap").hidden = state.period !== "custom";
          filtersChanged();
        });
      });
      document.querySelectorAll(".toolbar-section").forEach(function (details) {
        details.addEventListener("toggle", function () {
          if (!details.open) return;
          document.querySelectorAll(".toolbar-section[open]").forEach(function (other) {
            if (other !== details) other.open = false;
          });
        });
      });
      $("reset-filters").addEventListener("click", function () {
        Object.keys(state.selected).forEach(function (key) { state.selected[key] = []; });
        state.period = "month"; state.compare = "previous"; state.dateFrom = ""; state.dateTo = "";
        state.search = "";
        state.findingCode = "";
        state.sortBy = "date";
        state.sortDir = "desc";
        $("period").value = state.period; $("compare").value = state.compare;
        $("case-search").value = "";
        $("sort-by").value = "date";
        $("sort-dir").value = "desc";
        $("date-from").value = ""; $("date-to").value = "";
        $("date-from-wrap").hidden = true; $("date-to-wrap").hidden = true;
        document.querySelectorAll(".filter-pop").forEach(renderFilter);
        $("filters-panel").open = false;
        filtersChanged();
        showToast("Фильтры сброшены");
      });
      $("save-view").addEventListener("click", saveView);
      $("analysis-back").addEventListener("click", function () {
        if (!state.drillTrail.length) return;
        state.drillTrail.pop();
        if (!state.drillTrail.length) {
          clearDrillTrail(true);
          filtersChanged();
          return;
        }
        var last = state.drillTrail[state.drillTrail.length - 1];
        renderAnalysisRail();
        if (last && last.apply) last.apply();
      });
      $("analysis-clear").addEventListener("click", function () {
        clearDrillTrail(true);
        filtersChanged();
      });
      $("bulk-assign").addEventListener("click", function () {
        var assignee = prompt("Ответственный");
        if (assignee) bulkChange({ status: "assigned", assignee: assignee });
      });
      $("bulk-status").addEventListener("click", function () {
        bulkChange({ status: $("bulk-status-value").value });
      });
      $("queue-open-pdf-selected").addEventListener("click", function () {
        openSelectedQueuePdfs().catch(function (error) { showError(error.message); });
      });
      $("queue-critical-only").addEventListener("click", function () {
        state.findingCode = "";
        state.search = "";
        state.selected.statuses = ["critical"];
        $("case-search").value = "";
        showToast("Применён фильтр: только критические");
        filtersChanged();
      });
      $("yesterday-findings-list").addEventListener("click", function (event) {
        var caseButton = event.target.closest("[data-open-case]");
        if (caseButton) {
          event.preventDefault();
          openCase(caseButton.getAttribute("data-open-case"), caseButton);
          return;
        }
        var button = event.target.closest("[data-yesterday-finding]");
        if (button) {
          navigateYesterdayFinding(
            button.getAttribute("data-yesterday-finding"),
            button.getAttribute("data-yesterday-label"),
            button.getAttribute("data-yesterday-day")
          );
        }
      });
      document.addEventListener("click", function (event) {
        var pdfButton = event.target.closest("[data-open-pdf]");
        if (!pdfButton) return;
        event.preventDefault();
        event.stopPropagation();
        var popup = window.open("", "_blank");
        if (!popup) {
          showError("Разрешите всплывающие окна для открытия PDF.");
          return;
        }
        popup.document.write("<p style='font-family:Segoe UI,sans-serif;padding:18px'>Загрузка PDF...</p>");
        openPdfWithToken(pdfButton.getAttribute("data-open-pdf"), { preferredName: pdfButton.getAttribute("data-open-name"), targetWindow: popup })
          .catch(function (error) { showError(error.message); });
      });

      $("yesterday-action-rows").addEventListener("click", function (event) {
        var button = event.target.closest("[data-take-case]");
        if (button) {
          event.preventDefault();
          event.stopPropagation();
          takeYesterdayCase(button.getAttribute("data-take-case"), button);
        }
      });
      $("yesterday-flow-dimension").addEventListener("change", function () {
        if (state.data.daily) renderYesterdayFlow(state.data.daily, this.value);
      });
      $("saved-view").addEventListener("change", function () { if (this.value !== "") loadView(this.value); });
      $("view-manager").addEventListener("click", function (event) {
        var loadButton = event.target.closest("[data-load-view]");
        var deleteButton = event.target.closest("[data-delete-view]");
        if (loadButton) loadView(loadButton.getAttribute("data-load-view"));
        if (deleteButton) deleteView(deleteButton.getAttribute("data-delete-view"));
      });
      $("share-view").addEventListener("click", function () {
        navigator.clipboard.writeText(location.href); $("announcer").textContent = "Ссылка скопирована";
      });
      $("share-yesterday").addEventListener("click", function () {
        navigator.clipboard.writeText(location.href); $("announcer").textContent = "Ссылка скопирована";
      });
      $("print-report").addEventListener("click", function () { window.print(); });
      $("open-briefing-html").addEventListener("click", function () {
        var day = currentReportDay();
        window.open("/api/methodist/mo/briefing?format=html&date=" + encodeURIComponent(day), "_blank", "noopener");
      });
      $("theme-toggle").addEventListener("click", function () {
        var dark = this.getAttribute("aria-pressed") !== "true";
        try { localStorage.setItem(THEME_KEY, dark ? "dark" : "light"); } catch (error) {}
        applyPreferences();
        if (state.data.summary) renderOverview(state.data.summary);
        showToast(dark ? "Тёмная тема включена" : "Светлая тема включена");
      });
      $("density").addEventListener("change", function () {
        try { localStorage.setItem(DENSITY_KEY, this.value); } catch (error) {}
        applyPreferences();
        showToast(this.value === "compact" ? "Компактная плотность включена" : "Комфортная плотность включена");
      });
      $("methodology").addEventListener("change", function () {
        state.methodology = this.value === "v3" ? "v3" : "v4";
        state.data = {};
        showToast("Выбрана методика " + state.methodology);
        loadPage(state.page);
      });
      $("admin-token-save").addEventListener("click", function () {
        var value = $("admin-token-input").value.trim();
        if (!value) {
          showToast("Введите админ-токен");
          return;
        }
        try {
          sessionStorage.setItem(MO.api.ROLE_KEY, "admin");
          sessionStorage.setItem(MO.api.ADMIN_TOKEN_KEY, value);
        } catch (error) {
          showError("Не удалось сохранить админ-токен для сессии.");
          return;
        }
        $("admin-token-input").value = "";
        showToast("Роль администратора включена до закрытия вкладки");
      });
      $("command-open").addEventListener("click", function () { openCommandPalette(this); });
      $("command-backdrop").addEventListener("click", closeCommandPalette);
      $("command-search").addEventListener("input", function () { renderCommands(this.value); });
      $("command-search").addEventListener("keydown", function (event) {
        if (event.key === "ArrowDown") { event.preventDefault(); moveOption($("command-results"), 1); }
      });
      $("command-results").addEventListener("keydown", function (event) {
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          event.preventDefault(); moveOption(this, event.key === "ArrowDown" ? 1 : -1);
        } else if (event.key === "Enter") {
          event.preventDefault(); document.activeElement.click();
        }
      });
      var searchTimer = null;
      $("case-search").addEventListener("input", function () {
        var value = this.value;
        updateFilterSummary();
        window.clearTimeout(searchTimer);
        searchTimer = window.setTimeout(function () { renderSearchSuggestions(value); }, 250);
      });
      $("case-search").addEventListener("keydown", function (event) {
        if (event.key === "ArrowDown" && !$("search-suggestions").hidden) {
          event.preventDefault(); moveOption($("search-suggestions"), 1);
        } else if (event.key === "Escape") {
          $("search-suggestions").hidden = true; this.setAttribute("aria-expanded", "false");
        }
      });
      $("search-suggestions").addEventListener("keydown", function (event) {
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          event.preventDefault(); moveOption(this, event.key === "ArrowDown" ? 1 : -1);
        } else if (event.key === "Enter") {
          event.preventDefault(); document.activeElement.click();
        } else if (event.key === "Escape") {
          this.hidden = true; $("case-search").focus();
        }
      });
      $("drawer-close").addEventListener("click", closeDrawer); $("drawer-backdrop").addEventListener("click", closeDrawer);
      document.addEventListener("keydown", function (event) {
        if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
          event.preventDefault();
          if ($("command-palette").hidden) openCommandPalette(document.activeElement);
          else closeCommandPalette();
          return;
        }
        if (!$("command-palette").hidden) {
          if (event.key === "Escape") { event.preventDefault(); closeCommandPalette(); }
          if (event.key === "Tab") {
            var commandFocus = $("command-palette").querySelectorAll('button, input, [tabindex]:not([tabindex="-1"])');
            if (!commandFocus.length) return;
            var commandFirst = commandFocus[0], commandLast = commandFocus[commandFocus.length - 1];
            if (event.shiftKey && document.activeElement === commandFirst) { event.preventDefault(); commandLast.focus(); }
            else if (!event.shiftKey && document.activeElement === commandLast) { event.preventDefault(); commandFirst.focus(); }
          }
          return;
        }
        if ($("case-drawer").hidden) return;
        if (event.key === "Escape") closeDrawer();
        if (event.key === "Tab") {
          var focusable = $("case-drawer").querySelectorAll('button, a[href], select, input, [tabindex]:not([tabindex="-1"])');
          if (!focusable.length) return;
          var first = focusable[0], last = focusable[focusable.length - 1];
          if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
          else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
        }
      });
      $("token-submit").addEventListener("click", function () {
        var value = $("token-input").value.trim();
        if (!value) { $("auth-error").textContent = "Введите токен."; return; }
        try { localStorage.setItem(TOKEN_KEY, value); sessionStorage.setItem(TOKEN_KEY, value); } catch (e) {}
        setAuth(false); loadCapabilities(); loadPage(state.page);
      });
      window.addEventListener("popstate", function () { readUrl(); renderChips(); switchPage(state.page, false); });
      renderSavedViews(); refreshSavedViews();
    }
    async function init() {
      readUrl(); applyPreferences(); bind();
      if ($("methodology")) $("methodology").value = state.methodology === "v3" ? "v3" : "v4";
      renderChips();
      renderAnalysisRail();
      ensureColumnState();
      $("columns-manager").hidden = true;
      if (!token()) setAuth(true);
      else { setAuth(false); await loadCapabilities(); switchPage(state.page, false); }
    }
    MO.app = Object.freeze({ init: init, switchPage: switchPage, showToast: showToast });
    init();
  })(window.MO = window.MO || {});
