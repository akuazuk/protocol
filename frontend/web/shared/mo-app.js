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
      page: "overview", period: "month", compare: "previous", pageNo: 1, dateFrom: "", dateTo: "", search: "", findingCode: "",
      sortBy: "date", sortDir: "desc",
      selected: { months: [], branches: [], specialties: [], doctors: [], document_types: [], statuses: [] },
      data: {}, facets: {}, trigger: null, openCaseId: ""
    };
    var PAGE_TITLES = {
      overview: "Обзор МО", yesterday: "Отчёт за вчера", queue: "Очередь разбора",
      documents: "Все случаи", doctors: "Врачи", specialties: "Специальности",
      diagnoses: "Диагнозы и МКБ", safety: "Безопасность", "data-quality": "Качество данных",
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
      var today = minskDateKey(0);
      if (!state.selected.months.length) {
        if (state.period === "month") q.set("month", today.slice(0, 7));
        if (state.period === "yesterday") {
          q.set("date_from", minskDateKey(-1)); q.set("date_to", minskDateKey(-1));
        }
        if (state.period === "7d") {
          q.set("date_from", minskDateKey(-6)); q.set("date_to", today);
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
        coverage: agg.avg_coverage || raw.avg_coverage || null,
        confidence: agg.avg_confidence || raw.avg_confidence || null,
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
    function score(value) { return value == null ? "Нет данных" : Math.round(Number(value)) + "%"; }
    function bar(name, value, suffix) {
      var n = Math.max(0, Math.min(100, Number(value) || 0));
      var cls = n < 60 ? " bad" : n < 75 ? " warn" : "";
      return '<div class="bar"><div class="bar-name" title="' + esc(name) + '">' + esc(name) +
        '</div><div class="track"><div class="fill' + cls + '" style="width:' + n + '%"></div></div><b>' +
        esc(suffix == null ? Math.round(n) + "%" : suffix) + "</b></div>";
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
    }
    function renderFilter(details) {
      var key = details.getAttribute("data-filter");
      var list = state.facets[key] || [];
      var selected = state.selected[key] || [];
      details.querySelector("summary b").textContent = selected.length ? selected.length : "Все";
      details.querySelector(".filter-menu").innerHTML =
        '<input class="control" type="search" placeholder="Найти" aria-label="Поиск по фильтру">' +
        '<div class="filter-options">' + list.map(function (item) {
          return '<label class="filter-option" data-label="' + esc(item.label.toLowerCase()) + '"><input type="checkbox" value="' +
            esc(item.value) + '"' + (selected.indexOf(item.value) >= 0 ? " checked" : "") + '><span>' +
            esc(item.label) + '</span><span class="filter-count">' + esc(item.count == null ? "" : item.count) + "</span></label>";
        }).join("") + "</div>";
      var search = details.querySelector('input[type="search"]');
      search.addEventListener("input", function () {
        var term = search.value.trim().toLowerCase();
        details.querySelectorAll(".filter-option").forEach(function (option) {
          option.hidden = option.getAttribute("data-label").indexOf(term) < 0;
        });
      });
      details.querySelectorAll('input[type="checkbox"]').forEach(function (input) {
        input.addEventListener("change", function () {
          var index = state.selected[key].indexOf(input.value);
          if (input.checked && index < 0) state.selected[key].push(input.value);
          if (!input.checked && index >= 0) state.selected[key].splice(index, 1);
          details.querySelector("summary b").textContent = state.selected[key].length || "Все";
          filtersChanged();
        });
      });
    }
    function renderChips() {
      var html = [];
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
      var clearFinding = $("filter-chips").querySelector("[data-clear-finding]");
      if (clearFinding) clearFinding.addEventListener("click", function () {
        state.findingCode = "";
        filtersChanged();
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
    }
    function filtersChanged() {
      state.pageNo = 1;
      renderChips(); syncUrl(true); loadPage(state.page);
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
      loadPage(page);
      $("main").focus({ preventScroll: true });
      window.scrollTo(0, 0);
    }
    function renderTrendChart(summary, monthly, specialties) {
      var points = monthly.length ? monthly.map(function (item) {
        return { label: item.month || "Месяц", value: item.avg_score != null ? item.avg_score : item.avg_overall };
      }) : specialties.length ? specialties.map(function (item) {
        return { label: item.specialization || item.specialty || "Группа", value: item.avg_overall_pct != null ? item.avg_overall_pct : item.avg_overall };
      }) : [{ label: "Итоговая оценка", value: summary.score || 0 }];
      var element = $("trend-chart");
      var option = {
        tooltip: { trigger: "axis", valueFormatter: function (value) { return Math.round(Number(value) || 0) + "%"; } },
        grid: { left: 45, right: 18, top: 18, bottom: 45 },
        xAxis: { type: "category", name: "Период", data: points.map(function (item) { return item.label; }) },
        yAxis: { type: "value", name: "Оценка, %", min: 0, max: 100 },
        series: [{ name: "Итоговая оценка", type: "line", smooth: true, areaStyle: { opacity: .16 },
          symbolSize: 8, data: points.map(function (item) { return item.value; }) }]
      };
      var chart = MO.moChart(element, option, {
        label: "Динамика оценки МО по выбранному периоду",
        description: "Линейный график итоговой оценки в процентах.",
        fallback: function (target) {
          target.innerHTML = points.map(function (item) { return bar(item.label, item.value); }).join("");
        }
      });
      if (chart) chart.on("click", function () { switchPage("documents"); });
    }
    function renderOverview(summary) {
      state.data.summary = summary;
      $("overview-kpis").innerHTML =
        kpi("МО за период", summary.n.toLocaleString("ru-RU"), "записи из БД МИС") +
        kpi("Оценено", summary.evaluated.toLocaleString("ru-RU"), "доля от допустимых к оценке") +
        kpi("Итоговая оценка", score(summary.score), "по всем оценённым") +
        kpi("Требует внимания", summary.attention || "0", (summary.attentionPct || 0) + "% выборки") +
        kpi("Критические", summary.critical || "0", "приоритетный ручной разбор") +
        kpi("Полнота проверки", score(summary.coverage), "доступность данных") +
        kpi("Надёжность", score(summary.confidence), "устойчивость результата") +
        kpi("Свежесть данных", summary.generated ? "Актуально" : "Не указана", summary.generated || "время не указано");
      var freshness = summary.raw.data_freshness || {};
      var freshnessBadge = $("freshness");
      freshnessBadge.classList.remove("stale", "critical");
      if (freshness.status === "critical") freshnessBadge.classList.add("critical");
      else if (freshness.status === "stale") freshnessBadge.classList.add("stale");
      var lagText = freshness.lag_days == null ? "без даты" : (freshness.lag_days + " дн.");
      freshnessBadge.textContent = summary.generated
        ? "Данные обновлены " + summary.generated + " (" + lagText + ")"
        : "Свежесть не указана";
      var empty = freshness.empty_state || {};
      if (empty.reason_code && empty.reason_code !== "ok") {
        var critical = empty.reason_code === "no_source_data" || freshness.status === "critical";
        $("global-error").className = critical ? "banner critical" : "banner";
        showError(empty.title + ". " + (empty.hint || ""));
      }
      var specialties = summary.specialties.slice(0, 8);
      var monthly = ((summary.raw.trends || {}).monthly || []).filter(function (x) {
        return !state.selected.months.length || state.selected.months.indexOf(x.month) >= 0;
      });
      renderTrendChart(summary, monthly, specialties);
      $("reason-chart").innerHTML = summary.findings.length ? summary.findings.slice(0, 7).map(function (x) {
        var name = x.title || x.label || x.code || "Замечание", pct = x.pct == null ? x.count || x.n || 0 : x.pct;
        return bar(name, Math.min(100, pct), x.pct == null ? String(pct) : Math.round(pct) + "%");
      }).join("") : '<div class="empty">Причины появятся после обработки данных.</div>';
      $("diagnosis-chart").innerHTML = summary.diagnoses.length ? summary.diagnoses.slice(0, 7).map(function (x) {
        return bar(x.chapter || x.label || x.mkb_chapter || "Без кода", x.avg_overall || x.avg_overall_pct || 0);
      }).join("") : '<div class="empty">Нет данных по главам МКБ.</div>';
      $("attention-list").innerHTML =
        notice("Критические", (summary.critical || 0) + " случаев требуют первоочередного решения", "critical") +
        notice("На разбор", (summary.attention || 0) + " случаев в рабочей очереди", "review") +
        notice("Качество данных", summary.generated ? "Последняя загрузка завершена" : "Проверьте свежесть загрузки", summary.generated ? "good" : "review");
      buildFacets(summary, summary.raw.facets);
    }
    async function loadOverview() {
      var suffix = "?" + query().toString();
      var responses = await Promise.all([
        request("/overview" + suffix, "__root__"),
        request("/facets" + suffix, "/cases" + suffix),
        request("/trends" + suffix, "/dynamics" + suffix)
      ]);
      var response = responses[0], facetsResponse = responses[1], trendsResponse = responses[2];
      if (response.status === 401 || response.status === 403) { setAuth(true); return; }
      if (!response.ok) throw new Error("Не удалось загрузить обзор.");
      var raw = await response.json();
      if (facetsResponse.ok) {
        var facetPayload = await facetsResponse.json();
        raw.facets = facetPayload.facets || facetPayload;
      }
      if (trendsResponse.ok) raw.trends = await trendsResponse.json();
      renderOverview(normalizeSummary(raw));
    }
    function rowRecord(row) {
      var id = row.case_id || row.visit_id || row.id || "";
      var doctor = row.doctor_fio || row.doctor || "Врач не указан";
      var specialty = row.specialization || row.specialty || "";
      var diagnosis = row.diagnosis_short || row.diagnosis || row.mkb_code_main || "Не указан";
      var total = row.deep_overall_pct != null ? row.deep_overall_pct : (row.overall_pct != null ? row.overall_pct : row.l1_overall_pct);
      var status = row.crm_status || ((row.crm || {}).status) || row.deep_status || row.status || (total < 60 ? "critical" : total < 75 ? "review" : "good");
      return { raw: row, id: id, date: row.date || row.visit_date || "", doctor: doctor, specialty: specialty,
        branch: row.filial || row.branch || "", diagnosis: diagnosis, total: total, status: status,
        kind: row.document_kind_label || row.kz_kind_label || row.kz_kind || "Не указан",
        coverage: row.coverage_pct || row.coverage, confidence: row.confidence_pct || row.confidence };
    }
    function statusLabel(value) {
      var map = { new:"Новый", assigned:"Назначен", in_review:"На разборе", confirmed_issue:"Подтверждено",
        false_positive:"Отклонено", needs_more_data:"Нужны данные", sent_to_doctor:"Передано врачу",
        resolved:"Решено", closed:"Закрыто", critical:"Критично", review:"Требует внимания",
        poor:"Требует внимания", good:"Хорошо", acceptable:"Приемлемо", needs_review:"Требует внимания",
        case_action:"Решение по случаю", bulk_action:"Групповое изменение" };
      return map[value] || value || "Не указан";
    }
    function statusClass(value) { return /critical|confirmed/.test(value) ? "critical" : /good|resolved|closed|acceptable/.test(value) ? "good" : "review"; }
    function documentRow(item) {
      return '<tr tabindex="0" data-case="' + esc(item.id) + '"><td>' + esc(item.date) + '</td><td><b>' + esc(item.doctor) +
        '</b><br><small>' + esc(item.specialty) + '</small></td><td>' + esc(item.branch) + '</td><td>' + esc(item.diagnosis) +
        '</td><td>' + esc(item.kind) + '</td><td><b>' + esc(score(item.total)) + '</b></td><td>' + esc(score(item.coverage)) +
        '</td><td>' + esc(score(item.confidence)) + '</td><td><span class="status ' + statusClass(item.status) + '">' +
        esc(statusLabel(item.status)) + "</span></td></tr>";
    }
    function queueRow(item) {
      var priority = Number(item.raw.p0 || 0) > 0 ? "P0" : Number(item.raw.p1 || 0) > 0 ? "P1" : "Низкий балл";
      var crm = item.raw.crm || {};
      return '<tr tabindex="0" data-case="' + esc(item.id) + '"><td><input type="checkbox" data-case-select="' + esc(item.id) + '" aria-label="Выбрать случай"></td><td><span class="status ' +
        statusClass(item.status) + '">' + esc(priority) + '</span></td><td>' + esc(item.date) +
        '</td><td>' + esc(item.branch) + '</td><td><b>' + esc(item.doctor) + '</b><br><small>' + esc(item.specialty) +
        '</small></td><td>' + esc(item.diagnosis) + '</td><td>' + esc(score(item.total)) + '</td><td>' +
        esc(item.raw.reason || item.raw.comment || "Требует ручной проверки") + '</td><td>' +
        esc(item.raw.assignee || crm.assignee || "Не назначен") + '</td><td>' + esc(item.raw.due_date || crm.due_date || "Сегодня") +
        '</td><td>' + esc(statusLabel(item.status)) + "</td></tr>";
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
        '<tr><td colspan="' + (queue ? 11 : 9) + '" class="empty"><b>' +
        esc(emptyState.title || "По выбранным фильтрам случаев нет.") + "</b><div>" +
        esc(emptyState.hint || "Измените фильтры или расширьте период.") + "</div></td></tr>";
      bindCaseRows(body);
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
      $("drawer-body").innerHTML =
        '<div class="drawer-grid">' + kpi("Итоговая оценка", score(data.deep_overall_pct != null ? data.deep_overall_pct : item.total), "по доступным данным") +
        kpi("Статус", statusLabel(data.deep_status || item.status), "рабочий статус") +
        kpi("Полнота проверки", score(data.coverage_pct || item.coverage), "доступность исходных данных") +
        kpi("Надёжность", score(data.confidence_pct || item.confidence), "устойчивость результата") + '</div>' +
        '<div class="detail-block"><h3>Оси оценки</h3>' + ["documentation","clinical_concordance","safety","regulatory"].map(function (key) {
          var labels = { documentation:"Оформление", clinical_concordance:"Согласованность", safety:"Безопасность", regulatory:"Регуляторика" };
          return bar(labels[key], axes[key] == null ? record["axis_" + key] : axes[key]);
        }).join("") + '</div>' +
        '<div class="detail-block"><h3>Клиническая цепочка</h3><p>' + esc(item.diagnosis) +
        '</p><p class="card-sub">Жалобы и анамнез → статус → диагноз → МКБ → обследования → лечение → наблюдение</p></div>' +
        '<div class="detail-block"><h3>Выявленные замечания</h3>' + (findings.length ? findings.map(function (finding) {
          var title = [finding.code, finding.title_ru || finding.title].filter(Boolean).join(" · ");
          var decision = (crm.finding_decisions || {})[finding.code] || "unreviewed";
          return notice(finding.severity || "Проверить", title || finding.detail_ru || finding.detail || "Требуется ручная проверка",
            finding.severity === "P0" ? "critical" : "review") +
            ((finding.detail_ru || finding.detail) ? '<p>' + esc(finding.detail_ru || finding.detail) + '</p>' : "") +
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
        '<p><button class="button" id="drawer-save" type="button">Сохранить решение</button> <a class="button secondary" href="/doctor/review?source=mo&amp;case=' + encodeURIComponent(item.id) + '">Анализ документа</a></p></div>' +
        '<div class="detail-block"><h3>История решений</h3>' + (events.length ? events.map(function (event) {
          return notice(new Date(event.created_at).toLocaleString("ru-RU"), statusLabel(event.event_type) + " · " + (event.actor || "методист"), "good");
        }).join("") : '<p class="empty">Решений пока нет.</p>') + '</div>';
      $("drawer-save").addEventListener("click", saveCaseDecision);
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
      if (!completeness.available) {
        $("yesterday-completeness").innerHTML = unavailableBlock(completeness);
      } else {
        var expected = completeness.expected_rows || {};
        $("yesterday-completeness").innerHTML =
          kpi("Получено", completeness.actual_rows, "строк из источника") +
          kpi("Ожидалось", expected.available ? expected.value : "Нет базы", expected.available ? expected.samples + " сопоставимых дней" : expected.reason) +
          notice("Лаг", completeness.lag_days + " дн. · ревизия " + (completeness.revision == null ? "не указана" : completeness.revision),
            completeness.partial ? "День помечен как неполный" : "День завершён", completeness.partial ? "critical" : "good") +
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
          { name: "За день", type: "bar", data: available.map(function (item) { return item.value; }) },
          { name: "Предыдущий день", type: "bar", data: available.map(function (item) { return item.previous_day; }) },
          { name: "Среднее дня недели", type: "line", symbolSize: 9, data: available.map(function (item) { return item.weekday_mean_8w; }) }
        ]
      }, {
        label: "Сравнение четырёх индексов за вчера",
        description: "Для каждого индекса показано значение дня, предыдущего дня и среднее того же дня недели.",
        fallback: function (target) {
          target.innerHTML = available.map(function (item) { return bar(item.label, item.value); }).join("");
        }
      });
      if (chart) chart.on("click", function () { switchPage("documents"); });
    }
    function navigateFinding(code) {
      state.findingCode = code || "";
      state.search = "";
      $("case-search").value = "";
      switchPage("documents");
    }
    function renderYesterdayFindings(data) {
      var items = ((data.top_findings || {}).items || []).slice(0, 12);
      if (!items.length) {
        $("yesterday-findings-chart").innerHTML = unavailableBlock(data.top_findings);
        $("yesterday-findings-list").innerHTML = "";
        return;
      }
      var total = items.reduce(function (sum, item) { return sum + Number(item.cases || 0); }, 0), running = 0;
      var cumulative = items.map(function (item) { running += Number(item.cases || 0); return total ? Math.round(1000 * running / total) / 10 : 0; });
      var chart = MO.moChart($("yesterday-findings-chart"), {
        tooltip: { trigger: "axis" },
        grid: { left: 48, right: 48, top: 30, bottom: 95 },
        xAxis: { type: "category", name: "Замечание", axisLabel: { rotate: 35 }, data: items.map(function (item) { return item.finding_code; }) },
        yAxis: [{ type: "value", name: "Случаи" }, { type: "value", name: "Накоплено, %", min: 0, max: 100 }],
        series: [
          { name: "Случаи", type: "bar", data: items.map(function (item) {
            return { value: item.cases, itemStyle: { decal: { symbol: item.severity === "P0" ? "rect" : "line" } } };
          }) },
          { name: "Накопленная доля", type: "line", yAxisIndex: 1, data: cumulative }
        ]
      }, {
        label: "Парето замечаний за вчера",
        description: "Столбцы показывают число случаев, линия - накопленную долю.",
        fallback: function (target) {
          target.innerHTML = items.map(function (item) { return bar(item.finding_code, Math.min(100, item.cases), item.cases); }).join("");
        }
      });
      if (chart) chart.on("click", function (params) { navigateFinding(items[params.dataIndex].finding_code); });
      $("yesterday-findings-list").innerHTML = items.map(function (item) {
        return '<button class="finding-link" type="button" data-yesterday-finding="' + esc(item.finding_code) +
          '"><b>' + esc(item.severity) + "</b> " + esc(item.label) + " · " + esc(item.cases) + " случаев</button>";
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
        state.selected.doctors = [doctor]; renderChips(); switchPage("documents");
      });
      $("yesterday-doctor-note").innerHTML = '<p class="inline-note">' + esc(section.rule) + "</p>";
    }
    function renderYesterdayActions(data) {
      var section = data.action_cases || {}, items = section.items || [];
      $("yesterday-action-rows").innerHTML = items.length ? items.map(function (item) {
        return '<tr data-case="' + esc(item.case_id) + '"><td><span class="status ' +
          (item.severity === "P0" ? "critical" : "review") + '">' + esc(item.severity) +
          "</span></td><td><b>" + esc(item.doctor) + "</b><br><small>" + esc(item.specialty) +
          "</small></td><td>" + esc(item.branch) + "</td><td>" + esc(item.diagnosis) +
          "</td><td><b>" + esc(item.finding_code) + "</b><br><small>" + esc(item.reason) +
          '</small></td><td><button class="button secondary" type="button" data-take-case="' +
          esc(item.case_id) + '"' + (item.crm_status === "in_review" ? " disabled" : "") + ">" +
          (item.crm_status === "in_review" ? "Уже в работе" : "Взять в работу") + "</button></td></tr>";
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
        state.selected[stateKey] = [items[params.dataIndex].key]; renderChips(); switchPage("documents");
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
      var yesterday = minskDateKey(-1);
      $("yesterday-date").textContent = "Итоги за " + new Date(yesterday + "T12:00:00").toLocaleDateString("ru-RU", { dateStyle:"long" }) + ".";
      var response = await request("/daily-report?date=" + yesterday, "__root__");
      if (response.status === 401 || response.status === 403) { setAuth(true); return; }
      if (!response.ok) throw new Error("Отчёт за вчера пока недоступен.");
      var data = await response.json();
      state.data.daily = data;
      $("partial-banner").hidden = !(data.partial || data.quality_status === "blocked");
      renderYesterday(data);
    }
    function renderEntityPages(summary) {
      $("doctor-rows").innerHTML = summary.doctors.length ? summary.doctors.slice(0,100).map(function (x) {
        return "<tr><td><b>" + esc(x.doctor_fio || x.doctor) + "</b></td><td>" + esc(x.specialization || x.specialty) +
          "</td><td>" + esc(x.n || 0) + "</td><td>" + esc(score(x.avg_overall_pct || x.avg_overall)) +
          "</td><td>" + esc(x.bad_pct || 0) + "%</td><td>" + esc(x.open_cases || 0) + "</td></tr>";
      }).join("") : '<tr><td colspan="6" class="empty">Нет данных по врачам.</td></tr>';
      var specialtyBars = summary.specialties.slice(0,20).map(function (x) {
        return bar(x.specialization || x.specialty, x.avg_overall_pct || x.avg_overall || 0);
      }).join("");
      $("specialty-chart").innerHTML = specialtyBars || '<div class="empty">Нет данных.</div>';
      $("specialty-attention").innerHTML = summary.specialties.slice(0,5).map(function (x) {
        return notice(x.specialization || x.specialty, (x.n || 0) + " записей, итог " + score(x.avg_overall_pct || x.avg_overall), "review");
      }).join("") || '<div class="empty">Нет групп внимания.</div>';
      $("icd-chart").innerHTML = summary.diagnoses.slice(0,20).map(function (x) {
        return bar(x.chapter || x.label || "Без кода", x.avg_overall || x.avg_overall_pct || 0);
      }).join("") || '<div class="empty">Нет данных.</div>';
      $("diagnosis-findings").innerHTML = summary.findings.slice(0,8).map(function (x) {
        return notice(x.severity || "Проверить", x.title || x.label || "Требуется ручная проверка", x.severity === "P0" ? "critical" : "review");
      }).join("") || '<div class="empty">Замечаний по выбранному срезу нет.</div>';
      $("safety-kpis").innerHTML = kpi("Критические", summary.critical, "потенциальный вред") + kpi("На разбор", summary.attention, "ручная проверка") +
        kpi("Надёжность", score(summary.confidence), "результата") + kpi("Полнота", score(summary.coverage), "проверки");
      $("safety-list").innerHTML = $("diagnosis-findings").innerHTML;
      $("quality-kpis").innerHTML = kpi("Свежесть", summary.generated ? "Актуально" : "Нет отметки", summary.generated) +
        kpi("Записей", summary.n, "получено") + kpi("Оценено", summary.evaluated, "после проверки") + kpi("Пропуски", Math.max(0, summary.n-summary.evaluated), "не допущено");
      $("quality-chart").innerHTML = bar("Обработано", summary.n ? summary.evaluated / summary.n * 100 : 0) +
        bar("Полнота проверки", summary.coverage || 0) + bar("Надёжность", summary.confidence || 0);
      $("quality-warnings").innerHTML = notice(summary.generated ? "Загрузка завершена" : "Нет времени обновления",
        summary.generated || "Проверьте источник данных", summary.generated ? "good" : "review");
      $("report-list").innerHTML = notice("Отчёт за вчера", "Готов к открытию и выгрузке", "good") +
        notice("Отчёт за текущий месяц", "Формируется по текущему срезу", "review");
    }
    async function ensureSummary() {
      if (!state.data.summary) await loadOverview();
      renderEntityPages(state.data.summary);
    }
    async function loadDataQuality() {
      var response = await request("/data-quality?" + query().toString(), "/cases?" + query().toString());
      if (!response.ok) throw new Error("Не удалось загрузить качество данных.");
      var data = await response.json();
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
    }
    async function loadReports() {
      var responses = await Promise.all([request("/reports", "/dynamics"), request("/freshness?" + query().toString(), "/dynamics")]);
      var response = responses[0], freshnessResponse = responses[1];
      if (!response.ok) throw new Error("Не удалось загрузить список отчётов.");
      var data = await response.json(), items = data.items || [];
      var freshness = freshnessResponse.ok ? await freshnessResponse.json() : (data.freshness || {});
      var health = "";
      if (freshness && freshness.status) {
        var statusText = freshness.status === "fresh" ? "Актуально" : freshness.status === "stale" ? "Требует обновления" : "Нет свежих данных";
        health = notice(
          "Состояние загрузки",
          statusText + " · лаг " + (freshness.lag_days == null ? "не определён" : freshness.lag_days + " дн.") +
            " · данные до " + (freshness.data_through || "не определено"),
          freshness.status === "critical" ? "critical" : freshness.status === "stale" ? "review" : "good"
        );
      }
      $("report-list").innerHTML = health + (items.length ? items.map(function (item) {
        return notice(item.date || item.month || "Отчёт", "Ревизия " + (item.revision || 1) + " · " + statusLabel(item.quality_status || "готов"), item.quality_status === "blocked" ? "critical" : "good");
      }).join("") : '<div class="empty">Готовых отчётов пока нет.</div>');
    }
    async function loadScoringMethod() {
      await ensureSummary();
      var response = await request("/scoring-method", "/scoring-info");
      if (!response.ok) throw new Error("Не удалось загрузить методику оценки.");
      var data = await response.json();
      var axes = (data.axes || []).map(function (axis) {
        return notice(axis.label || axis.key, axis.desc || "", "good");
      }).join("");
      var gates = (data.risk_gate || []).map(function (rule) {
        return "<li>" + esc(rule) + "</li>";
      }).join("");
      $("scoring-method").innerHTML =
        '<p><b>Итог:</b> ' + esc(data.overall_rule || "") + '</p><div class="grid"><div class="span-6">' +
        axes + '</div><div class="span-6"><h3>Правила клинического риска</h3><ol>' + gates +
        '</ol><p class="card-sub">Пороговые значения: хорошо ' + esc((data.thresholds || {}).good) +
        ', приемлемо ' + esc((data.thresholds || {}).acceptable) + '.</p></div></div>';
    }
    async function loadPage(page) {
      $("global-error").hidden = true;
      try {
        if (page === "overview") await loadOverview();
        else if (page === "yesterday") await loadYesterday();
        else if (page === "queue") await loadCases(true);
        else if (page === "documents") await loadCases(false);
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
        return '<div style="display:flex;justify-content:space-between;gap:8px;padding:9px 0;border-bottom:1px solid var(--line)"><button class="button secondary" data-load-view="' +
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
    function loadView(key) {
      var view = viewByKey(key); if (!view) return;
      var suffix = view.url;
      if (!suffix) {
        var params = new URLSearchParams(view.filters || {});
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
          state.search = $("case-search").value;
          box.hidden = true;
          $("case-search").setAttribute("aria-expanded", "false");
          filtersChanged();
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
        var table = document.querySelector("#page-documents table");
        var compact = table.classList.toggle("compact-columns");
        this.setAttribute("aria-pressed", compact ? "true" : "false");
        this.textContent = compact ? "Показать все колонки" : "Скрыть служебные колонки";
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
      $("case-search").addEventListener("change", function () { state.search = this.value.trim(); filtersChanged(); });
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
        document.querySelectorAll(".filter-pop").forEach(renderFilter); filtersChanged();
      });
      $("save-view").addEventListener("click", saveView);
      $("bulk-assign").addEventListener("click", function () {
        var assignee = prompt("Ответственный");
        if (assignee) bulkChange({ status: "assigned", assignee: assignee });
      });
      $("bulk-status").addEventListener("click", function () {
        bulkChange({ status: $("bulk-status-value").value });
      });
      $("yesterday-findings-list").addEventListener("click", function (event) {
        var button = event.target.closest("[data-yesterday-finding]");
        if (button) navigateFinding(button.getAttribute("data-yesterday-finding"));
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
        setAuth(false); loadPage(state.page);
      });
      window.addEventListener("popstate", function () { readUrl(); renderChips(); switchPage(state.page, false); });
      renderSavedViews(); refreshSavedViews();
    }
    async function init() {
      readUrl(); applyPreferences(); bind(); renderChips();
      if (!token()) setAuth(true);
      else { setAuth(false); switchPage(state.page, false); }
    }
    MO.app = Object.freeze({ init: init, switchPage: switchPage, showToast: showToast });
    init();
  })(window.MO = window.MO || {});
