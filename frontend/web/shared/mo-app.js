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
    var isExpertMode = function () { return !!(MO.api.isExpertAudience && MO.api.isExpertAudience()); };
    var hasSession = function () {
      if (isExpertMode()) return !!(MO.api.expertToken && MO.api.expertToken());
      if (MO.api.appSessionToken && MO.api.appSessionToken()) return true;
      return !!token();
    };
    var EXPERT_PAGES = { yesterday: true, reports: true };
    var state = {
      page: "yesterday", period: "yesterday", compare: "previous", methodology: "v4", pageNo: 1, dateFrom: "", dateTo: "", search: "", findingCode: "", rubricCriterion: "",
      reg55Band: "", reg55Pack: "", icdVisitStatus: "",
      sortBy: "date", sortDir: "desc",
      zoneFilter: "", zoneBandFilter: "", attentionOnly: false, shadowAttentionOnly: false, kpStatus: "", historyTier: "",
      worstSeverity: "",
      doctorZoneMetric: "zone1",
      caseNavIds: [],
      protocolSuggest: null,
      selected: { months: [], branches: [], specialties: [], doctors: [], document_types: ["clinical_visit"], statuses: [] },
      scoreEligibleOnly: true,
      data: {}, facets: {}, trigger: null, openCaseId: "", cabinetDoctorKey: "",
      caseDetail: null, supersedesPackId: "",
      drillTrail: [], drillSnapshot: null,
      columnVisible: { documents: [], queue: [] }, columnsPanelOpen: false,
      expertDisplayName: ""
    };
    var ZONE_LABELS = { zone1: "Оформление", zone2a: "Диагноз", zone2b: "План по протоколу" };
    var ZONE_PRESETS = {
      dx: { name: "Внимание: диагноз", zoneFilter: "zone2a", zoneBandFilter: "bad", attentionOnly: true, page: "documents" },
      plan: { name: "Внимание: план по КП", zoneFilter: "zone2b", zoneBandFilter: "bad", attentionOnly: true, kpStatus: "matched", page: "documents" },
      docs: { name: "Оформление слабо", zoneFilter: "zone1", zoneBandFilter: "bad", attentionOnly: false, page: "documents" },
      "first-plan": { name: "Первый контакт + слабый план", zoneFilter: "zone2b", zoneBandFilter: "bad", historyTier: "first_contact", attentionOnly: false, page: "documents" }
    };
    var PAGE_TITLES = {
      overview: "Период", yesterday: "Сегодня", queue: "Очередь",
      documents: "Все случаи", doctors: "Врачи",
      reports: "Отчёты", "kp-sync": "Протоколы МЗ", settings: "Справка"
    };
    var REMOVED_PAGES = {
      specialties: true, diagnoses: true, safety: true,
      "data-quality": true, "doctor-cabinet": true, "access-log": true
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
      if ($("density")) $("density").value = density;
      var dark = theme === "dark" || (!theme && window.matchMedia("(prefers-color-scheme: dark)").matches);
      if ($("theme-toggle")) {
        $("theme-toggle").setAttribute("aria-pressed", dark ? "true" : "false");
        $("theme-toggle").setAttribute("aria-label", dark ? "Включить светлую тему" : "Включить тёмную тему");
      }
    }
    function esc(value) {
      return String(value == null ? "" : value).replace(/&/g, "&amp;").replace(/</g, "&lt;")
        .replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    }
    var tableChromeState = {};
    var TABLE_FILTER_MAX_OPTIONS = 40;
    function cellPlainText(node) {
      return String((node && (node.innerText || node.textContent)) || "").replace(/\s+/g, " ").trim();
    }
    function rowLooksBad(tr) {
      if (!tr) return false;
      if (tr.querySelector(".status.critical, .status.important, .zone-band--bad, .band-bad, .zone-card--bad")) return true;
      var text = cellPlainText(tr).toLowerCase();
      return text.indexOf("плохо") >= 0 || text.indexOf("критич") >= 0 || text.indexOf("important") >= 0;
    }
    function columnIsUtility(th, colIndex, sampleRows) {
      var label = cellPlainText(th).toLowerCase();
      if (!label || label === "открыть" || label === "мо" || label.indexOf("выбрать") >= 0) return true;
      if (th.querySelector('input[type="checkbox"]')) return true;
      var utility = 0;
      var checked = 0;
      sampleRows.slice(0, 8).forEach(function (tr) {
        var td = tr.cells[colIndex];
        if (!td) return;
        checked += 1;
        var onlyControl = td.querySelector("button, input, a") && cellPlainText(td).length < 3;
        if (onlyControl || !cellPlainText(td)) utility += 1;
      });
      return checked > 0 && utility >= checked;
    }
    function uniqueColumnValues(rows, colIndex) {
      var counts = {};
      rows.forEach(function (tr) {
        var td = tr.cells[colIndex];
        if (!td) return;
        var value = cellPlainText(td);
        if (!value || value === "-") return;
        if (value.length > 64) value = value.slice(0, 61) + "...";
        counts[value] = (counts[value] || 0) + 1;
      });
      return Object.keys(counts).sort(function (a, b) {
        return counts[b] - counts[a] || a.localeCompare(b, "ru");
      });
    }
    function parseSortValue(text) {
      var raw = String(text || "").trim();
      if (!raw || raw === "-") return { num: null, str: "" };
      var pct = raw.replace(",", ".").match(/-?\d+(?:\.\d+)?/);
      if (pct && (/%|п\.п|плохо|норм/.test(raw.toLowerCase()) || /^[\d\s.,+-]+%?$/.test(raw))) {
        return { num: Number(pct[0]), str: raw.toLowerCase() };
      }
      if (/^\d{4}-\d{2}-\d{2}/.test(raw)) return { num: Date.parse(raw.slice(0, 10)), str: raw.toLowerCase() };
      if (/^\d+$/.test(raw)) return { num: Number(raw), str: raw };
      return { num: null, str: raw.toLowerCase() };
    }
    function compareCells(aText, bText, dir) {
      var a = parseSortValue(aText);
      var b = parseSortValue(bText);
      var sign = dir === "desc" ? -1 : 1;
      if (a.num != null && b.num != null && !isNaN(a.num) && !isNaN(b.num) && a.num !== b.num) {
        return (a.num - b.num) * sign;
      }
      return a.str.localeCompare(b.str, "ru", { numeric: true, sensitivity: "base" }) * sign;
    }
    function applyTableChrome(table) {
      var id = table && table.getAttribute("data-table-chrome");
      if (!id || !tableChromeState[id]) return;
      var st = tableChromeState[id];
      var body = table.tBodies[0];
      if (!body) return;
      var rows = Array.prototype.slice.call(body.rows || []).filter(function (tr) {
        return !tr.querySelector("td.empty, td[colspan]") || tr.cells.length > 1;
      });
      var dataRows = Array.prototype.slice.call(body.rows || []).filter(function (tr) {
        return tr.cells.length && !tr.querySelector("td[colspan]");
      });
      if (st.sortCol != null && dataRows.length) {
        dataRows.sort(function (ra, rb) {
          return compareCells(
            cellPlainText(ra.cells[st.sortCol]),
            cellPlainText(rb.cells[st.sortCol]),
            st.sortDir
          );
        });
        dataRows.forEach(function (tr) { body.appendChild(tr); });
      }
      var visible = 0;
      Array.prototype.forEach.call(body.rows || [], function (tr) {
        if (tr.querySelector("td[colspan]")) {
          tr.classList.remove("table-row-hidden");
          return;
        }
        var show = true;
        if (st.chip === "bad" && !rowLooksBad(tr)) show = false;
        if (show && st.search) {
          if (cellPlainText(tr).toLowerCase().indexOf(st.search) < 0) show = false;
        }
        if (show && st.colFilters) {
          Object.keys(st.colFilters).forEach(function (idx) {
            var want = st.colFilters[idx];
            if (!want) return;
            var got = cellPlainText(tr.cells[Number(idx)]);
            if (got.length > 64) got = got.slice(0, 61) + "...";
            if (got !== want) show = false;
          });
        }
        tr.classList.toggle("table-row-hidden", !show);
        if (show) visible += 1;
      });
      if (st.metaEl) {
        var total = dataRows.length || rows.length;
        st.metaEl.textContent = "Показано " + visible + " из " + total;
      }
      table.querySelectorAll("thead tr:first-child th").forEach(function (th) {
        var col = Number(th.getAttribute("data-col-index"));
        var key = th.getAttribute("data-sort-key") || "";
        if (st.serverSort && key && key.indexOf("col:") !== 0) {
          th.setAttribute("aria-sort", state.sortBy === key ? (state.sortDir === "asc" ? "ascending" : "descending") : "none");
        } else if (st.sortCol === col) {
          th.setAttribute("aria-sort", st.sortDir === "asc" ? "ascending" : "descending");
        } else {
          th.setAttribute("aria-sort", "none");
        }
      });
    }
    function attachTableChrome(table, options) {
      options = options || {};
      if (!table || !table.tHead || !table.tBodies || !table.tBodies[0]) return null;
      var wrap = table.closest(".table-wrap") || table.parentElement;
      if (!wrap || !wrap.parentElement) return null;
      var host = wrap.parentElement;
      var id = options.id || table.getAttribute("data-table-chrome") || ("tbl-" + Math.random().toString(36).slice(2, 8));
      table.setAttribute("data-table-chrome", id);
      var headerCells = Array.prototype.slice.call(table.tHead.rows[0].cells || []);
      var sampleRows = Array.prototype.slice.call(table.tBodies[0].rows || []).filter(function (tr) {
        return tr.cells.length && !tr.querySelector("td[colspan]");
      });
      var st = tableChromeState[id] || {
        search: "",
        chip: "all",
        colFilters: {},
        sortCol: null,
        sortDir: "asc",
        serverSort: !!options.serverSort,
        clientSort: !options.serverSort,
        metaEl: null
      };
      st.serverSort = !!options.serverSort;
      st.clientSort = !options.serverSort;
      tableChromeState[id] = st;

      var toolbar = host.querySelector('[data-table-toolbar="' + id + '"]');
      if (!toolbar) {
        toolbar = document.createElement("div");
        toolbar.className = "table-toolbar";
        toolbar.setAttribute("data-table-toolbar", id);
        host.insertBefore(toolbar, wrap);
      }
      toolbar.innerHTML =
        '<label class="filter"><span>Поиск в таблице</span>' +
        '<input class="control" type="search" data-table-search placeholder="Текст строки" autocomplete="off"></label>' +
        '<div class="table-toolbar-chips" role="group" aria-label="Быстрый фильтр">' +
        '<button type="button" class="chip-btn" data-chip="all" aria-pressed="true">Все</button>' +
        '<button type="button" class="chip-btn" data-chip="bad" aria-pressed="false">Только плохо</button>' +
        "</div>" +
        '<div class="table-toolbar-meta" data-table-meta></div>';
      st.metaEl = toolbar.querySelector("[data-table-meta]");
      var searchInput = toolbar.querySelector("[data-table-search]");
      searchInput.value = st.search || "";
      searchInput.addEventListener("input", function () {
        st.search = String(searchInput.value || "").trim().toLowerCase();
        applyTableChrome(table);
      });
      toolbar.querySelectorAll("[data-chip]").forEach(function (btn) {
        btn.setAttribute("aria-pressed", btn.getAttribute("data-chip") === st.chip ? "true" : "false");
        btn.addEventListener("click", function () {
          st.chip = btn.getAttribute("data-chip") || "all";
          toolbar.querySelectorAll("[data-chip]").forEach(function (other) {
            other.setAttribute("aria-pressed", other.getAttribute("data-chip") === st.chip ? "true" : "false");
          });
          applyTableChrome(table);
        });
      });

      var filterRow = table.tHead.querySelector("tr.col-filters");
      if (filterRow) filterRow.remove();
      filterRow = document.createElement("tr");
      filterRow.className = "col-filters";
      headerCells.forEach(function (th, colIndex) {
        th.setAttribute("data-col-index", String(colIndex));
        if (!th.classList.contains("sortable-th")) th.classList.add("sortable-th");
        if (!th.getAttribute("data-sort-key")) th.setAttribute("data-sort-key", "col:" + colIndex);
        var cell = document.createElement("th");
        cell.scope = "col";
        if (columnIsUtility(th, colIndex, sampleRows)) {
          cell.innerHTML = '<span class="sr-only">Без фильтра</span>';
          filterRow.appendChild(cell);
          return;
        }
        var values = uniqueColumnValues(sampleRows, colIndex);
        var select = document.createElement("select");
        select.setAttribute("data-col-filter", String(colIndex));
        select.setAttribute("aria-label", "Фильтр: " + cellPlainText(th));
        var allOpt = document.createElement("option");
        allOpt.value = "";
        allOpt.textContent = "Все";
        select.appendChild(allOpt);
        if (!values.length) {
          select.disabled = true;
        } else {
          values.slice(0, TABLE_FILTER_MAX_OPTIONS).forEach(function (value) {
            var opt = document.createElement("option");
            opt.value = value;
            opt.textContent = value;
            select.appendChild(opt);
          });
          if (values.length > TABLE_FILTER_MAX_OPTIONS) {
            var more = document.createElement("option");
            more.disabled = true;
            more.textContent = "… ещё " + (values.length - TABLE_FILTER_MAX_OPTIONS) + " (поиск)";
            select.appendChild(more);
          }
        }
        if (st.colFilters[colIndex]) select.value = st.colFilters[colIndex];
        select.addEventListener("change", function () {
          if (select.value) st.colFilters[colIndex] = select.value;
          else delete st.colFilters[colIndex];
          applyTableChrome(table);
        });
        cell.appendChild(select);
        filterRow.appendChild(cell);
      });
      table.tHead.appendChild(filterRow);

      headerCells.forEach(function (th) {
        if (th.__moSortBound) return;
        th.__moSortBound = true;
        th.addEventListener("click", function (event) {
          if (event.target && event.target.closest("select, input, button, a, .col-filters")) return;
          var col = Number(th.getAttribute("data-col-index"));
          var key = th.getAttribute("data-sort-key") || "";
          if (st.serverSort && key && key.indexOf("col:") !== 0) {
            if (state.sortBy === key) state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
            else {
              state.sortBy = key;
              state.sortDir = key === "date" ? "desc" : "asc";
            }
            state.pageNo = 1;
            st.sortCol = null;
            if ($("sort-by")) $("sort-by").value = state.sortBy;
            if ($("sort-dir")) $("sort-dir").value = state.sortDir;
            filtersChanged();
            return;
          }
          if (st.sortCol === col) st.sortDir = st.sortDir === "asc" ? "desc" : "asc";
          else {
            st.sortCol = col;
            st.sortDir = "asc";
          }
          applyTableChrome(table);
        });
      });
      applyTableChrome(table);
      return id;
    }
    function enhanceTablesIn(root, options) {
      if (!root) return;
      var tables = root.tagName === "TABLE" ? [root] : root.querySelectorAll("table");
      Array.prototype.forEach.call(tables, function (table, index) {
        var opts = Object.assign({}, options || {});
        if (!opts.id) {
          var body = table.tBodies[0];
          opts.id = (body && body.id) ? ("chrome-" + body.id) : ((options && options.idPrefix || "chrome") + "-" + index);
        }
        attachTableChrome(table, opts);
      });
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
      var searchRaw = String(state.search || "").trim();
      var patientMatch = searchRaw.match(/^patient(?:_id)?\s*[:=]\s*(\d+)\s*$/i);
      var visitMatch = searchRaw.match(/^visit(?:_id)?\s*[:=]\s*(\d+)\s*$/i);
      var idLookup = !!(patientMatch || visitMatch || /^\d{4,}$/.test(searchRaw));
      if (!idLookup && !state.selected.months.length) {
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
      if (patientMatch) q.set("patient_id", patientMatch[1]);
      else if (visitMatch) q.set("visit_id", visitMatch[1]);
      else if (searchRaw) q.set("q", searchRaw);
      if (state.findingCode) q.set("finding_codes", state.findingCode);
      if (state.rubricCriterion) q.set("reg55_point", state.rubricCriterion);
      if (state.reg55Band) q.set("reg55_band", state.reg55Band);
      if (state.reg55Pack) q.set("reg55_pack", state.reg55Pack);
      q.set("sort_by", state.sortBy);
      q.set("sort_dir", state.sortDir);
      Object.keys(state.selected).forEach(function (key) {
        // `|` - не запятая: адреса филиалов содержат "," и ломали split на API
        if (state.selected[key].length) q.set(API_FILTER_KEYS[key] || key, state.selected[key].join("|"));
      });
      if (state.selected.months.length) q.set("month", state.selected.months[0]);
      // Жёстко: non-clinical вне таблицы; URL не даёт opt-out.
      q.set("score_eligible_only", "1");
      if (state.zoneFilter) q.set("zone", state.zoneFilter);
      if (state.zoneBandFilter) q.set("zone_band", state.zoneBandFilter);
      if (state.attentionOnly) q.set("attention_only", "1");
      if (state.shadowAttentionOnly) q.set("shadow_attention_only", "1");
      if (state.kpStatus) q.set("kp_status", state.kpStatus);
      if (state.historyTier) q.set("history_tier", state.historyTier);
      if (state.icdVisitStatus) q.set("icd_visit_status", state.icdVisitStatus);
      if (state.worstSeverity) q.set("worst_severity", state.worstSeverity);
      return q;
    }
    function applyScoreEligibleOnly(on, silent) {
      // Жёстко: только clinical_visit в таблице; non-clinical не оцениваем и не показываем.
      state.scoreEligibleOnly = true;
      state.selected.document_types = ["clinical_visit"];
      var toggle = $("score-eligible-only");
      if (toggle) {
        toggle.checked = true;
        toggle.disabled = true;
      }
      var filter = document.querySelector('.filter-pop[data-filter="document_types"]');
      if (filter && filter.querySelector(".filter-menu")) {
        renderFilter(filter);
        var summaryB = filter.querySelector("summary b");
        if (summaryB) summaryB.textContent = "Клинический приём";
      }
      if (!silent) filtersChanged();
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
    function shouldForceReauth(status, detail) {
      if (status === 401) return true;
      if (status !== 403) return false;
      if (!hasSession()) return true;
      var text = String(detail || "").toLowerCase();
      // Permission-style 403: keep the session, show an in-app error.
      if (
        text.indexOf("роль") >= 0 ||
        text.indexOf("недоступ") >= 0 ||
        text.indexOf("недостаточно прав") >= 0 ||
        text.indexOf("администратор") >= 0 ||
        text.indexOf("только к") >= 0 ||
        text.indexOf("только администратору") >= 0
      ) {
        return false;
      }
      // Auth-style 403: invalid/missing token or expired session.
      if (
        text.indexOf("token") >= 0 ||
        text.indexOf("токен") >= 0 ||
        text.indexOf("сессия") >= 0 ||
        text.indexOf("methodist") >= 0 ||
        text.indexOf("логин") >= 0
      ) {
        return true;
      }
      // Default: do not kick a live session on an ambiguous 403.
      return false;
    }
    async function handleHttpAuth(response) {
      if (!response || (response.status !== 401 && response.status !== 403)) return false;
      var detail = "";
      try {
        var data = await response.clone().json();
        if (typeof data.detail === "string") detail = data.detail;
        else if (data.detail != null) detail = JSON.stringify(data.detail);
      } catch (error) {}
      if (shouldForceReauth(response.status, detail)) {
        if (isExpertMode() && MO.api.clearExpertToken) MO.api.clearExpertToken();
        else if (MO.api.appSessionToken && MO.api.appSessionToken()) {
          if (MO.api.clearAppSessionToken) MO.api.clearAppSessionToken();
        } else if (MO.api.clearToken) MO.api.clearToken();
        setAuth(true, detail || "Требуется повторный вход.");
        return true;
      }
      showError(detail || "Недостаточно прав для этого раздела.");
      return true;
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
        document_types: values(rawFacets.document_types || rawFacets.document_kinds || rawFacets.kz_kind || ["clinical_visit","procedure_session","medical_exam","diagnostic","non_clinical"], ["value"]),
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
      details.classList.toggle("has-applied", selected.length > 0);
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
        if (key === "document_types") {
          // Не даём выбрать процедуры/профосмотры в таблице случаев.
          draft = ["clinical_visit"];
          state.selected.document_types = ["clinical_visit"];
          selected = ["clinical_visit"];
          state.scoreEligibleOnly = true;
          details.querySelector("summary b").textContent = "Клинический приём";
          if ($("score-eligible-only")) {
            $("score-eligible-only").checked = true;
            $("score-eligible-only").disabled = true;
          }
        }
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
        if (options.zoneFilter !== undefined) state.zoneFilter = options.zoneFilter;
        if (options.zoneBandFilter !== undefined) state.zoneBandFilter = options.zoneBandFilter;
        if (options.attentionOnly !== undefined) state.attentionOnly = !!options.attentionOnly;
        if (options.kpStatus !== undefined) state.kpStatus = options.kpStatus || "";
        if (options.historyTier !== undefined) state.historyTier = options.historyTier || "";
        if (options.worstSeverity !== undefined) state.worstSeverity = options.worstSeverity || "";
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
        html.push('<span class="chip">№55 пункт: ' + esc(state.rubricCriterion) +
          '<button type="button" data-clear-rubric aria-label="Удалить фильтр пункта №55">×</button></span>');
      }
      if (state.reg55Band) {
        html.push('<span class="chip">Градация №55: ' + esc(reg55BandLabelRu(state.reg55Band)) +
          '<button type="button" data-clear-reg55-band aria-label="Удалить фильтр градации №55">×</button></span>');
      }
      if (state.reg55Pack) {
        html.push('<span class="chip">Pack №55: ' + esc(state.reg55Pack) +
          '<button type="button" data-clear-reg55-pack aria-label="Удалить фильтр pack №55">×</button></span>');
      }
      if (state.zoneFilter || state.zoneBandFilter) {
        html.push('<span class="chip">Раздел: ' +
          esc((ZONE_LABELS[state.zoneFilter] || state.zoneFilter || "любой") +
            (state.zoneBandFilter ? " · " + state.zoneBandFilter : "")) +
          '<button type="button" data-clear-zone aria-label="Удалить фильтр раздела">×</button></span>');
      }
      if (state.attentionOnly) {
        html.push('<span class="chip">Только внимание<button type="button" data-clear-attention aria-label="Снять фильтр внимания">×</button></span>');
      }
      if (state.kpStatus) {
        html.push('<span class="chip">КП: ' + esc(state.kpStatus) +
          '<button type="button" data-clear-kp aria-label="Удалить фильтр КП">×</button></span>');
      }
      if (state.historyTier) {
        html.push('<span class="chip">История: ' + esc(state.historyTier) +
          '<button type="button" data-clear-history-tier aria-label="Удалить фильтр истории">×</button></span>');
      }
      if (state.icdVisitStatus) {
        html.push('<span class="chip">МКБ: ' + esc(state.icdVisitStatus) +
          '<button type="button" data-clear-icd-status aria-label="Удалить фильтр МКБ">×</button></span>');
      }
      if (state.worstSeverity) {
        var sevChip = ({ P0: "Критично", P1: "Важно", P2: "Умеренно", P3: "Оформление" })[state.worstSeverity] || state.worstSeverity;
        html.push('<span class="chip">Худший уровень: ' + esc(sevChip) +
          '<button type="button" data-clear-worst-severity aria-label="Сбросить фильтр приоритета">×</button></span>');
      }
      $("filter-chips").innerHTML = html.join("");
      $("filter-chips").querySelectorAll("[data-remove]").forEach(function (button) {
        button.addEventListener("click", function () {
          var key = button.getAttribute("data-remove"), value = button.getAttribute("data-value");
          state.selected[key] = state.selected[key].filter(function (x) { return x !== value; });
          if (key === "document_types") {
            applyScoreEligibleOnly(true, true);
          }
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
        filtersChanged();
      });
      var clearReg55Band = $("filter-chips").querySelector("[data-clear-reg55-band]");
      if (clearReg55Band) clearReg55Band.addEventListener("click", function () {
        state.reg55Band = "";
        filtersChanged();
      });
      var clearReg55Pack = $("filter-chips").querySelector("[data-clear-reg55-pack]");
      if (clearReg55Pack) clearReg55Pack.addEventListener("click", function () {
        state.reg55Pack = "";
        filtersChanged();
      });
      var clearZone = $("filter-chips").querySelector("[data-clear-zone]");
      if (clearZone) clearZone.addEventListener("click", function () {
        state.zoneFilter = "";
        state.zoneBandFilter = "";
        filtersChanged();
      });
      var clearAttention = $("filter-chips").querySelector("[data-clear-attention]");
      if (clearAttention) clearAttention.addEventListener("click", function () {
        state.attentionOnly = false;
        filtersChanged();
      });
      var clearKp = $("filter-chips").querySelector("[data-clear-kp]");
      if (clearKp) clearKp.addEventListener("click", function () {
        state.kpStatus = "";
        filtersChanged();
      });
      var clearHistoryTier = $("filter-chips").querySelector("[data-clear-history-tier]");
      if (clearHistoryTier) clearHistoryTier.addEventListener("click", function () {
        state.historyTier = "";
        filtersChanged();
      });
      var clearIcd = $("filter-chips").querySelector("[data-clear-icd-status]");
      if (clearIcd) clearIcd.addEventListener("click", function () {
        state.icdVisitStatus = "";
        filtersChanged();
      });
      var clearWorst = $("filter-chips").querySelector("[data-clear-worst-severity]");
      if (clearWorst) clearWorst.addEventListener("click", function () {
        state.worstSeverity = "";
        filtersChanged();
      });
    }
    function syncUrl(replace) {
      var q = query();
      q.set("page", state.page);
      var path;
      if (isExpertMode()) {
        path = state.page === "reports" ? "/methodist/expert/reports" : "/methodist/expert/yesterday";
      } else {
        path = state.page === "yesterday" ? "/methodist/mo/yesterday" :
          (state.page === "queue" || state.page === "documents" ? "/methodist/mo/cases" : "/methodist/mo");
      }
      var url = path + "?" + q.toString();
      history[replace ? "replaceState" : "pushState"]({ page: state.page }, "", url);
    }
    function readUrl() {
      var q = new URLSearchParams(location.search);
      var pathPage;
      if (isExpertMode()) {
        pathPage = location.pathname.indexOf("/reports") >= 0 ? "reports" : "yesterday";
      } else {
        pathPage = location.pathname.endsWith("/yesterday") ? "yesterday" :
          (location.pathname.endsWith("/cases") ? "documents" : "overview");
      }
      state.page = PAGE_TITLES[q.get("page")] ? q.get("page") : pathPage;
      if (isExpertMode() && !EXPERT_PAGES[state.page]) state.page = "yesterday";
      state.period = q.get("period") || (isExpertMode() ? "yesterday" : "month");
      state.compare = q.get("compare_period") || "previous";
      state.dateFrom = q.get("date_from") || ""; state.dateTo = q.get("date_to") || "";
      state.search = q.get("q") || "";
      state.findingCode = q.get("finding_codes") || "";
      state.rubricCriterion = q.get("reg55_point") || "";
      state.reg55Band = q.get("reg55_band") || "";
      state.reg55Pack = q.get("reg55_pack") || "";
      state.sortBy = q.get("sort_by") || "date";
      state.sortDir = q.get("sort_dir") || "desc";
      state.zoneFilter = q.get("zone") || "";
      state.zoneBandFilter = q.get("zone_band") || "";
      state.attentionOnly = q.get("attention_only") === "1" || q.get("attention_only") === "true";
      state.shadowAttentionOnly = q.get("shadow_attention_only") === "1" || q.get("shadow_attention_only") === "true";
      state.kpStatus = q.get("kp_status") || "";
      state.historyTier = q.get("history_tier") || "";
      state.icdVisitStatus = q.get("icd_visit_status") || "";
      state.worstSeverity = (q.get("worst_severity") || "").toUpperCase();
      Object.keys(state.selected).forEach(function (key) {
        state.selected[key] = (q.get(API_FILTER_KEYS[key] || key) || "").split(/[|,]/).filter(Boolean);
      });
      // URL score_eligible_only=0 / чужие document_types игнорируем.
      applyScoreEligibleOnly(true, true);
      if ($("period")) $("period").value = state.period;
      if ($("compare")) $("compare").value = state.compare;
      if ($("date-from")) $("date-from").value = state.dateFrom;
      if ($("date-to")) $("date-to").value = state.dateTo;
      if ($("case-search")) $("case-search").value = state.search;
      if ($("sort-by")) $("sort-by").value = state.sortBy;
      if ($("sort-dir")) $("sort-dir").value = state.sortDir;
      if ($("date-from-wrap")) $("date-from-wrap").hidden = state.period !== "custom";
      if ($("date-to-wrap")) $("date-to-wrap").hidden = state.period !== "custom";
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
      if (REMOVED_PAGES[page]) page = (page === "access-log" || page === "data-quality") ? "reports" : "documents";
      if (!PAGE_TITLES[page]) page = "yesterday";
      if (isExpertMode() && !EXPERT_PAGES[page]) page = "yesterday";
      state.page = page;
      document.querySelectorAll(".page").forEach(function (section) { section.hidden = section.getAttribute("data-page") !== page; });
      document.querySelectorAll(".nav-button").forEach(function (button) {
        if (button.getAttribute("data-page") === page) button.setAttribute("aria-current", "page");
        else button.removeAttribute("aria-current");
      });
      var helpBtn = $("sidebar-help");
      if (helpBtn) {
        if (page === "settings") helpBtn.setAttribute("aria-current", "page");
        else helpBtn.removeAttribute("aria-current");
      }
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
    function hostActive(id) {
      var el = $(id);
      return !!(el && !el.hasAttribute("hidden"));
    }
    function renderMonthTrend(data) {
      if (!hostActive("month-trend-chart")) return;
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
      if (!hostActive("month-heatmap-chart")) return;
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
      if (!hostActive("month-doctor-chart")) return;
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
      if (!hostActive("month-pareto-chart")) return;
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
      if (!hostActive("month-funnel-chart") && !hostActive("month-crm-chart")) return;
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
    function renderAttentionStrip(hostId, attention, opts) {
      var host = $(hostId);
      if (!host) return;
      opts = opts || {};
      var a = attention || {};
      if (!a || (!a.n_evaluated && a.n_evaluated !== 0)) {
        host.innerHTML = '<p class="card-sub">Оценки зон ещё не посчитаны за период (нужен recompute после деплоя движка).</p>';
        return;
      }
      function tile(label, value, meta, go, tone) {
        return '<button type="button" class="attention-tile attention-tile--' + esc(tone || "neutral") + '" data-attention-go="' + esc(go || "") + '">' +
          '<div class="kpi-label">' + esc(label) + '</div>' +
          '<div class="kpi-value">' + esc(value == null ? "-" : value) + '</div>' +
          (meta ? '<div class="kpi-meta">' + esc(meta) + '</div>' : "") +
          '</button>';
      }
      host.innerHTML =
        tile("Критично в очереди", a.queue_critical != null ? a.queue_critical : "-", "открыть очередь", "queue:critical", "critical") +
        tile("Важно в очереди", a.queue_important != null ? a.queue_important : "-", "открыть очередь", "queue:important", "important") +
        tile("Оформление плохо", a.zone1_bad, (a.zone1_bad_pct != null ? a.zone1_bad_pct + "%" : ""), "zone1:bad", "zone1") +
        tile("Диагноз плохо", a.zone2a_bad, (a.zone2a_bad_pct != null ? a.zone2a_bad_pct + "%" : ""), "zone2a:bad", "zone2a") +
        tile("План плохо", a.zone2b_bad, (a.zone2b_bad_pct != null ? a.zone2b_bad_pct + "%" : ""), "zone2b:bad", "zone2b");
      host.querySelectorAll("[data-attention-go]").forEach(function (btn) {
        btn.addEventListener("click", function () {
          var go = btn.getAttribute("data-attention-go") || "";
          if (go.indexOf("queue:") === 0) {
            switchPage("queue");
            return;
          }
          if (go.indexOf("zone") === 0) {
            var parts = go.split(":");
            state.zoneFilter = parts[0];
            state.zoneBandFilter = parts[1] || "bad";
            switchPage("documents");
          }
        });
      });
    }
    function cssToken(name, fallback) {
      try {
        var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
        return v || fallback;
      } catch (e) { return fallback; }
    }
    function renderZoneTrendHost(hostId, trends) {
      var host = $(hostId);
      if (!host) return;
      trends = (trends || []).slice(-14);
      if (!trends.length) {
        host.innerHTML = '<p class="empty">Нет тренда зон за период.</p>';
        return;
      }
      host.classList.add("zone-trend-chart");
      var dates = trends.map(function (row) { return row.date; });
      var c1 = cssToken("--zone-1", cssToken("--chart-1", "#2f6f63"));
      var c2 = cssToken("--zone-2a", cssToken("--chart-2", "#4a6fa5"));
      var c3 = cssToken("--zone-2b", cssToken("--chart-3", "#8a7a5a"));
      function series(name, key, color) {
        return {
          name: name,
          type: "line",
          smooth: true,
          showSymbol: trends.length <= 10,
          symbolSize: 7,
          lineStyle: { width: 2.4, color: color },
          itemStyle: { color: color },
          areaStyle: { color: color, opacity: 0.08 },
          data: trends.map(function (row) {
            var v = row[key];
            return v == null ? null : Number(v);
          })
        };
      }
      MO.moChart(host, {
        color: [c1, c2, c3],
        legend: { top: 4, data: ["Оформление", "Диагноз", "План"] },
        grid: { left: 42, right: 18, top: 42, bottom: 28 },
        tooltip: { trigger: "axis" },
        xAxis: { type: "category", data: dates, boundaryGap: false },
        yAxis: { type: "value", min: 0, max: 100, name: "%" },
        series: [
          series("Оформление", "zone1_avg", c1),
          series("Диагноз", "zone2a_avg", c2),
          series("План", "zone2b_avg", c3)
        ]
      }, {
        label: "Тренд трёх оценок",
        description: "Средние доли оформления, диагноза и плана по дням периода."
      });
    }
    function analyticsWindowLabel(win) {
      win = win || {};
      if (state.period === "yesterday") {
        return "вчера · динамика 14 дней до " + (win.trend_date_to || win.date_to || "…");
      }
      if (state.period === "7d") return "последние 7 дней";
      if (state.period === "month") return "текущий месяц (по дням)";
      if (win.date_from && win.date_to) {
        return win.date_from === win.date_to
          ? win.date_from
          : (win.date_from + " - " + win.date_to);
      }
      return "выбранный период";
    }
    function openZoneBandCases(zoneKey, band) {
      applyDrill({
        label: (ZONE_LABELS[zoneKey] || zoneKey) + " · " + (band || ""),
        zoneFilter: zoneKey || "",
        zoneBandFilter: band || "",
        attentionOnly: false,
        page: "documents"
      });
    }
    function openReg55BandCases(band) {
      pushDrill("№55 · " + reg55BandLabelRu(band), function () {
        state.reg55Band = band || "";
        state.pageNo = 1;
        renderChips();
        switchPage("documents");
        filtersChanged();
      });
      state.reg55Band = band || "";
      state.pageNo = 1;
      renderChips();
      switchPage("documents");
      filtersChanged();
    }
    function renderYesterdayScoreKpis(daily, dash) {
      var host = $("yesterday-score-kpis");
      if (!host) return;
      var funnel = (daily && daily.funnel) || {};
      var evaluated = funnel.evaluated;
      if (evaluated == null && daily && daily.attention) evaluated = daily.attention.n_evaluated;
      var eligible = funnel.eligible;
      var source = funnel.source != null ? funnel.source : eligible;
      var cov = null;
      if (eligible != null && Number(eligible) > 0 && evaluated != null) {
        cov = Math.round(1000 * Number(evaluated) / Number(eligible)) / 10;
      }
      var through = (daily && (daily.data_through || daily.date)) || "";
      var win = (dash && dash.window) || {};
      host.innerHTML =
        kpi("Получено", source == null ? "-" : source, "рабочий день") +
        kpi("Оценено", evaluated == null ? "-" : evaluated, "рабочий день") +
        kpi("Покрытие", cov == null ? "-" : (cov + "%"), "eligible") +
        kpi("Свежесть", through || "-", win.date_from
          ? ("кольца: " + analyticsWindowLabel(win))
          : "склад");
    }
    function renderScoreRing(card, title, centerText, segments, onSelect) {
      card.innerHTML =
        '<p class="score-ring-title">' + esc(title) + "</p>" +
        '<div class="score-ring-chart"></div>' +
        '<p class="score-ring-meta">' + esc(centerText || "") + "</p>";
      var chartHost = card.querySelector(".score-ring-chart");
      if (!MO.moDonut) {
        chartHost.innerHTML = '<p class="empty">Нет диаграмм</p>';
        return;
      }
      MO.moDonut(chartHost, segments, {
        centerText: String(centerText || "").split("\n")[0] || "-",
        label: title,
        description: "Распределение оценок. Клик по сегменту открывает случаи.",
        onSelect: onSelect,
        emptyText: "Нет оценок"
      });
    }
    function renderSeverityPriorityRing(counts) {
      var host = $("month-severity-ring");
      if (!host) return;
      counts = counts || {};
      var labels = { P0: "Критично", P1: "Важно", P2: "Умеренно", P3: "Оформление" };
      var colors = {
        P0: cssToken("--sev-p0", "#c0455a"),
        P1: cssToken("--sev-p1", "#c47830"),
        P2: cssToken("--sev-p2", "#8a7a3a"),
        P3: cssToken("--sev-p3", "#5b6f8f")
      };
      var segments = ["P0", "P1", "P2", "P3"].map(function (key) {
        return {
          key: key,
          band: key,
          name: labels[key],
          value: Number(counts[key] || 0),
          color: colors[key]
        };
      });
      var total = segments.reduce(function (sum, s) { return sum + (Number(s.value) || 0); }, 0);
      var noneN = Number(counts.none || 0);
      host.innerHTML =
        '<p class="score-ring-title">Худший уровень замечания</p>' +
        '<div class="score-ring-chart"></div>' +
        '<p class="score-ring-meta">' +
        (total ? ("с замечаниями: " + total + (noneN ? " · без: " + noneN : "")) : "Нет замечаний за период") +
        "</p>";
      var chartHost = host.querySelector(".score-ring-chart");
      if (!MO.moDonut) {
        chartHost.innerHTML = '<p class="empty">Нет диаграмм</p>';
        return;
      }
      MO.moDonut(chartHost, segments, {
        centerText: String(total || 0),
        centerSub: "случаев",
        label: "Приоритет замечаний",
        description: "Клик по сегменту открывает случаи с этим худшим уровнем замечания.",
        onSelect: function (key) {
          var page = (key === "P0" || key === "P1") ? "queue" : "documents";
          applyDrill({
            label: "Худший уровень: " + (labels[key] || key),
            worstSeverity: key,
            page: page
          });
        },
        emptyText: "Нет замечаний"
      });
    }
    function renderScoreRings(dash) {
      var host = $("yesterday-score-rings");
      if (!host) return;
      if (!dash || !dash.ok || !dash.available) {
        host.innerHTML = '<p class="empty">' +
          esc((dash && (dash.reason || dash.error)) || "Нет оценок за выбранное окно.") + "</p>";
        return;
      }
      var good = cssToken("--good", "#2f6f63");
      var warn = cssToken("--warn", "#9a7b3c");
      var bad = cssToken("--bad", "#9a5b66");
      var mute = cssToken("--muted", "#7a8494");
      var zones = dash.zones || {};
      var zoneMeta = [
        { key: "zone1", title: "Оформление" },
        { key: "zone2a", title: "Диагноз" },
        { key: "zone2b", title: "План по протоколу" }
      ];
      var zoneLabels = { ok: "в норме", weak: "слабо", bad: "плохо", na: "нет данных" };
      host.innerHTML = "";
      zoneMeta.forEach(function (meta) {
        var card = document.createElement("div");
        card.className = "score-ring";
        host.appendChild(card);
        var block = zones[meta.key] || {};
        var bands = block.bands || {};
        var segments = ["ok", "weak", "bad", "na"].map(function (band) {
          var row = bands[band] || {};
          return {
            band: band,
            name: zoneLabels[band],
            value: Number(row.n || 0),
            color: band === "ok" ? good : band === "weak" ? warn : band === "bad" ? bad : mute
          };
        });
        var center = block.avg_pct == null ? "-" : (Math.round(Number(block.avg_pct)) + "%");
        renderScoreRing(card, meta.title, center, segments, function (band) {
          openZoneBandCases(meta.key, band);
        });
      });
      var regCard = document.createElement("div");
      regCard.className = "score-ring";
      host.appendChild(regCard);
      var reg = dash.reg55 || {};
      if (!reg.available) {
        regCard.innerHTML =
          '<p class="score-ring-title">№55</p><p class="empty">№55 недоступен за окно</p>';
        return;
      }
      var share = reg.band_share || {};
      var regSeg = [
        { band: "compliant_min", name: "80-100%", color: good },
        { band: "compliant_measures", name: "55-79.9%", color: warn },
        { band: "noncompliant", name: "до 54.9%", color: bad },
        { band: "unscored", name: "не оценено", color: mute }
      ].map(function (s) {
        var row = share[s.band] || {};
        return { band: s.band, name: s.name, value: Number(row.n || 0), color: s.color };
      });
      var regCenter = reg.avg_pct == null ? "-" : (Math.round(Number(reg.avg_pct)) + "%");
      renderScoreRing(regCard, "№55", regCenter, regSeg, openReg55BandCases);
    }
    function renderScoreDynamics(dash) {
      var host = $("yesterday-score-dynamics");
      var sub = $("yesterday-dynamics-sub");
      if (!host) return;
      var trends = (dash && dash.trends) || [];
      var win = (dash && dash.window) || {};
      if (sub) {
        var from = win.trend_date_from || win.date_from || "";
        var to = win.trend_date_to || win.date_to || "";
        sub.textContent = from && to
          ? ("Средние % по дням: " + from + " - " + to + " · клик по дню открывает этот день")
          : "Средние % зон и №55 по дням выбранного периода";
      }
      if (!trends.length) {
        host.innerHTML = '<p class="empty">Нет динамики за окно аналитики.</p>';
        return;
      }
      host.classList.add("zone-trend-chart");
      var dates = trends.map(function (row) { return row.date; });
      var c1 = cssToken("--zone-1", cssToken("--chart-1", "#2f6f63"));
      var c2 = cssToken("--zone-2a", cssToken("--chart-2", "#4a6fa5"));
      var c3 = cssToken("--zone-2b", cssToken("--chart-3", "#8a7a5a"));
      var c55 = cssToken("--accent", "#3d5a80");
      function series(name, key, color, dashed) {
        return {
          name: name,
          type: "line",
          smooth: true,
          showSymbol: trends.length <= 14,
          symbolSize: 7,
          lineStyle: { width: dashed ? 2 : 2.4, color: color, type: dashed ? "dashed" : "solid" },
          itemStyle: { color: color },
          areaStyle: dashed ? undefined : { color: color, opacity: 0.07 },
          data: trends.map(function (row) {
            var v = row[key];
            return v == null ? null : Number(v);
          })
        };
      }
      var chart = MO.moChart(host, {
        color: [c1, c2, c3, c55],
        legend: { top: 4, data: ["Оформление", "Диагноз", "План", "№55"] },
        grid: { left: 42, right: 18, top: 42, bottom: 28 },
        tooltip: { trigger: "axis" },
        xAxis: { type: "category", data: dates, boundaryGap: false },
        yAxis: { type: "value", min: 0, max: 100, name: "%" },
        series: [
          series("Оформление", "zone1_avg", c1, false),
          series("Диагноз", "zone2a_avg", c2, false),
          series("План", "zone2b_avg", c3, false),
          series("№55", "reg55_avg", c55, true)
        ]
      }, {
        label: "Динамика оценок",
        description: "Средние проценты зон и №55 по дням окна аналитики."
      });
      if (chart) {
        chart.on("click", function (params) {
          var day = dates[params.dataIndex];
          if (!day) return;
          state.period = "custom";
          state.dateFrom = day;
          state.dateTo = day;
          if ($("period")) $("period").value = "custom";
          if ($("date-from")) $("date-from").value = day;
          if ($("date-to")) $("date-to").value = day;
          if ($("date-from-wrap")) $("date-from-wrap").hidden = false;
          if ($("date-to-wrap")) $("date-to-wrap").hidden = false;
          filtersChanged();
        });
      }
    }
    function renderYesterdayScoreDashboard(dash, workingDay) {
      var win = (dash && dash.window) || {};
      var analytics = $("yesterday-analytics-window");
      if (analytics) {
        analytics.textContent = "Рабочий день: " + (workingDay || "…") +
          " · в кольцах и динамике: " + analyticsWindowLabel(win);
      }
      renderScoreRings(dash);
      renderScoreDynamics(dash);
    }
    function renderOverview(data) {
      if (!data.available) { showError(data.reason || "Данные месяца недоступны."); return; }
      var summary=normalizeSummary(data), k=data.kpi || {}, forecast=data.forecast || {};
      state.data.summary=summary;
      $("month-period-label").textContent=(data.period_label || "Период")+" с "+data.period.date_from+" по "+data.data_through+
        ". Дней: "+data.days_elapsed+" из "+data.days_in_month+". Europe/Minsk.";
      $("freshness").textContent="Данные по "+data.data_through;
      $("month-kpis").innerHTML=kpi("Записи",k.source_records,"объём")+
        kpi("Оценено",k.evaluated,score(k.coverage_pct)+" покрытие")+
        kpi("Свежесть", $("freshness") ? $("freshness").textContent : "-", "склад")+
        kpi("Прогноз объёма",forecast.projected_source,"к концу месяца");
      // overview API may nest attention on month-report or separate overview call
      var attention = data.attention || (data.overview && data.overview.attention) || null;
      if (!attention && state.data.overviewAttention) attention = state.data.overviewAttention;
      renderAttentionStrip("month-attention", attention);
      renderSeverityPriorityRing(data.worst_severity_cases || {});
      renderZoneTrendHost("month-zone-trend", (attention && attention.zone_trends) || data.zone_trends || []);
      var look = $("month-look-where");
      if (look) {
        var docs = (data.by_doctor || []).slice().sort(function (a, b) {
          var av = Number(a.zone2a_bad_pct != null ? a.zone2a_bad_pct : a.bad_pct) || 0;
          var bv = Number(b.zone2a_bad_pct != null ? b.zone2a_bad_pct : b.bad_pct) || 0;
          return bv - av;
        }).slice(0, 8);
        look.innerHTML = docs.length ? '<div class="table-wrap"><table><thead><tr><th>Врач</th><th>Случаев</th><th>Оформл. плохо</th><th>Диагноз плохо</th><th>План плохо</th><th></th></tr></thead><tbody>' +
          docs.map(function (row) {
            var name = row.doctor_fio || row.doctor || "";
            return "<tr><td>" + esc(name) + "</td><td>" + esc(row.n) +
              "</td><td>" + esc(pctOrDash(row.zone1_bad_pct != null ? row.zone1_bad_pct : null)) +
              "</td><td>" + esc(pctOrDash(row.zone2a_bad_pct != null ? row.zone2a_bad_pct : null)) +
              "</td><td>" + esc(pctOrDash(row.zone2b_bad_pct != null ? row.zone2b_bad_pct : null)) +
              '</td><td><button type="button" class="button secondary compact" data-look-doctor="' +
              esc(name) + '">Открыть</button></td></tr>';
          }).join("") + "</tbody></table></div>" : '<p class="empty">Недостаточно данных.</p>';
        look.querySelectorAll("[data-look-doctor]").forEach(function (btn) {
          btn.addEventListener("click", function () {
            var name = btn.getAttribute("data-look-doctor") || "";
            if (!name) return;
            state.selected.doctors = [name];
            state.zoneBandFilter = "bad";
            switchPage("documents");
          });
        });
        enhanceTablesIn(look, { idPrefix: "chrome-month-look" });
      }
      if (hostActive("month-forecast")) {
        $("month-forecast").innerHTML=kpi("Прогноз записей",forecast.projected_source,forecast.method)+
          kpi("Прогноз оценённых",forecast.projected_evaluated,"при текущем темпе")+
          kpi("Прогноз оценки",score(forecast.projected_avg_score),"без изменения среднего");
      }
      if (hostActive("month-compare")) {
        var comparisons=data.comparison || {};
        $("month-compare").innerHTML=Object.keys(comparisons).map(function (key) {
          var item=comparisons[key];
          return item.available ? notice(item.label,
            "Записи "+signed(item.deltas.source_records,"")+"; оценка "+signed(item.deltas.avg_score),"good") :
            notice("Сравнение недоступно",item.reason,"review");
        }).join("")+"<p class=\"inline-note\">"+esc((forecast.assumptions || []).join(". "))+"</p>";
      }
      var reconciliation=data.reconciliation || {}, banner=$("month-reconciliation");
      banner.hidden=reconciliation.status === "ok";
      banner.className="banner critical";
      banner.textContent="Расхождение дневных и MTD итогов: источник "+reconciliation.source_delta+
        ", оценено "+reconciliation.evaluated_delta+". Данные не замаскированы.";
      renderMonthTrend(data);renderMonthHeatmap(data);renderMonthDoctors(data);renderMonthPareto(data);renderMonthFunnel(data);
      renderMonthReg55Section(data.reg55);
      renderMonthIcdStatus(data.icd_visit_status);
      renderMonthClinicalGaps(data.clinical_gaps, data.kp_unmatched);
    }
    function renderMonthReg55Section(reg55) {
      var hostKpi = $("month-reg55");
      var hostTable = $("month-rubric-mz");
      if (!reg55 || !reg55.available) {
        if (hostKpi) hostKpi.innerHTML = unavailableBlock(reg55, "Нет выборки №55 (разд. V) за период.");
        if (hostTable) hostTable.innerHTML = unavailableBlock(reg55, "Нет выборки для сводки пунктов №55.");
        return;
      }
      var share = reg55.band_share || {};
      function bandKpi(code, label) {
        var row = share[code] || {};
        return '<button type="button" class="kpi kpi--clickable" data-reg55-band="' + esc(code) + '">' +
          '<div class="kpi-label">' + esc(label) + "</div>" +
          '<div class="kpi-value">' + esc(row.n != null ? row.n : "-") + "</div>" +
          '<div class="kpi-meta">' + esc((row.pct != null ? row.pct + "%" : "") + " оценённых") +
          "</div></button>";
      }
      if (hostKpi) {
        hostKpi.innerHTML =
          kpi("Соответствие №55", score(reg55.avg_pct != null ? reg55.avg_pct : reg55.value),
            (reg55.reg55_band_label_ru || "п.12-13") + " · sample " + (reg55.sample_n || 0)) +
          bandKpi("compliant_min", "80-100%") +
          bandKpi("compliant_measures", "55-79,9%") +
          bandKpi("noncompliant", "≤54,9%");
      }
      if (!hostTable) return;
      var top = (reg55.top_fail || []).slice(0, 8).map(function (item) {
        return '<tr tabindex="0" data-rubric-criterion="' + esc(item.point || item.id) + '"><td><b>' +
          esc(item.point || item.id) + "</b> " + esc(item.title || "") + "</td><td>" + esc(item.zero_n) +
          "</td><td>" + esc(item.half_n) + "</td><td><b>" + esc(item.fail_pct) + "%</b></td></tr>";
      }).join("");
      var titles = {};
      (reg55.top_fail || []).forEach(function (item) { titles[item.point || item.id] = item.title || item.id; });
      var specialtyRows = (reg55.by_specialty || []).slice(0, 8).map(function (row) {
        var weak = (row.top_criteria || []).map(function (c) {
          return esc(titles[c.id] || c.id) + " (" + esc(c.fail_n) + ")";
        }).join("; ");
        return "<tr><td>" + esc(row.specialty) + "</td><td>" + esc(row.fail_n) +
          "</td><td>" + (weak || " - ") + "</td></tr>";
      }).join("");
      hostTable.innerHTML =
        kpi("Выборка", reg55.sample_n, (reg55.date_from || "") + " - " + (reg55.date_to || "")) +
        kpi("Оценено по №55", reg55.scored_n, "с применимыми пунктами") +
        '<div class="table-wrap" style="margin-top:10px"><table class="rubric-table"><thead><tr>' +
        "<th>Пункт разд. V</th><th>0</th><th>0.5</th><th>Доля слабостей</th></tr></thead><tbody>" +
        (top || '<tr><td colspan="4" class="empty">Слабых пунктов нет.</td></tr>') +
        "</tbody></table></div>" +
        '<p class="card-sub">Клик по пункту открывает очередь с фокусом этого критерия в карточке случая.</p>' +
        '<h3 style="margin:14px 0 8px;font-size:14px">Слабости по специальностям</h3>' +
        '<div class="table-wrap"><table class="rubric-table"><thead><tr>' +
        "<th>Специальность</th><th>Слабых оценок</th><th>Топ пункты</th></tr></thead><tbody>" +
        (specialtyRows || '<tr><td colspan="3" class="empty">Недостаточно данных по специальностям.</td></tr>') +
        "</tbody></table></div>";
      hostTable.querySelectorAll("[data-rubric-criterion]").forEach(function (row) {
        function openCriterion(event) {
          if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") return;
          event.preventDefault();
          state.rubricCriterion = row.getAttribute("data-rubric-criterion") || "";
          state.pageNo = 1;
          renderChips();
          switchPage("queue");
          filtersChanged();
        }
        row.addEventListener("click", openCriterion);
        row.addEventListener("keydown", openCriterion);
      });
      enhanceTablesIn(hostTable, { idPrefix: "chrome-month-rubric" });
      if (hostKpi) {
        hostKpi.querySelectorAll("[data-reg55-band]").forEach(function (btn) {
          btn.addEventListener("click", function () {
            state.reg55Band = btn.getAttribute("data-reg55-band") || "";
            state.pageNo = 1;
            renderChips();
            switchPage("queue");
            filtersChanged();
          });
        });
      }
    }

    function renderMonthIcdStatus(payload) {
      var host = $("month-icd-status");
      if (!host) return;
      if (!payload || !payload.available) {
        host.innerHTML = unavailableBlock(payload, "Сводка чипа МКБ недоступна.");
        return;
      }
      var counts = payload.counts || {};
      var order = ["missing_dx", "not_in_directory", "weak_name", "ok", "unknown"];
      host.innerHTML = order.map(function (st) {
        var row = counts[st] || {};
        return '<button type="button" class="kpi kpi--clickable" data-icd-status="' + esc(st) + '" title="' +
          esc(row.title_ru || "") + '"><div class="kpi-label">' + esc(row.label_ru || st) +
          '</div><div class="kpi-value">' + esc(row.n != null ? row.n : "-") +
          '</div><div class="kpi-meta">из ' + esc(payload.sample_n || 0) + '</div></button>';
      }).join("");
      host.querySelectorAll("[data-icd-status]").forEach(function (btn) {
        btn.addEventListener("click", function () {
          state.icdVisitStatus = btn.getAttribute("data-icd-status") || "";
          state.pageNo = 1;
          renderChips();
          switchPage("documents");
          filtersChanged();
        });
      });
    }
    function renderMonthClinicalGaps(payload, kpUnmatched) {
      var host = $("month-clinical-gaps");
      if (!host) return;
      var kp = kpUnmatched || {};
      var head = "";
      if (kp && kp.available && kp.n != null) {
        head = '<button type="button" class="kpi kpi--clickable" data-kp-unmatched="1">' +
          '<div class="kpi-label">' + esc(kp.label_ru || "План без КП") + "</div>" +
          '<div class="kpi-value">' + esc(kp.n) + "</div>" +
          '<div class="kpi-meta">протокол не подобран</div></button>';
      }
      if (!payload || !payload.available) {
        host.innerHTML = head + unavailableBlock(payload, "Нет клинических разрывов в выборке.");
      } else {
        var rows = (payload.items || []).slice(0, 8).map(function (item) {
          return '<tr tabindex="0" data-gap-code="' + esc(item.finding_code) + '"><td>' +
            esc(item.label || item.finding_code) + "</td><td><b>" + esc(item.cases) + "</b></td></tr>";
        }).join("");
        host.innerHTML = head +
          kpi("Случаев с разрывом", payload.cases_with_gaps, "клинические разрывы") +
          '<div class="table-wrap" style="margin-top:10px"><table class="rubric-table"><thead><tr>' +
          "<th>Разрыв</th><th>Случаев</th></tr></thead><tbody>" +
          (rows || '<tr><td colspan="2" class="empty">Пусто.</td></tr>') +
          "</tbody></table></div>";
        host.querySelectorAll("[data-gap-code]").forEach(function (row) {
          function openGap(event) {
            if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") return;
            event.preventDefault();
            state.findingCode = row.getAttribute("data-gap-code") || "";
            state.pageNo = 1;
            renderChips();
            switchPage("documents");
            filtersChanged();
          }
          row.addEventListener("click", openGap);
          row.addEventListener("keydown", openGap);
        });
        enhanceTablesIn(host, { idPrefix: "chrome-month-gaps" });
      }
      host.querySelectorAll("[data-kp-unmatched]").forEach(function (btn) {
        btn.addEventListener("click", function () {
          state.kpStatus = "unmatched";
          state.pageNo = 1;
          renderChips();
          switchPage("documents");
          filtersChanged();
        });
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
        request("/reg55-section-summary?" + rubricQuery.toString()),
        request("/overview" + suffix)
      ]);
      var response = responses[0], facetsResponse = responses[1], reg55Response = responses[2], overviewResponse = responses[3];
      if (await handleHttpAuth(response)) return;
      if (!response.ok) throw new Error("Не удалось загрузить отчёт месяца.");
      var raw = await response.json();
      if (facetsResponse.ok) {
        var facetPayload = await facetsResponse.json();
        raw.facets = facetPayload.facets || facetPayload;
      }
      // month-report уже кладёт reg55; summary API - fallback / обогащение top_fail
      if (reg55Response && reg55Response.ok) {
        var reg55Payload = await reg55Response.json();
        if (reg55Payload && reg55Payload.available) {
          raw.reg55 = Object.assign({}, raw.reg55 || {}, reg55Payload, { available: true });
        } else if (!(raw.reg55 && raw.reg55.available)) {
          raw.reg55 = reg55Payload || { available: false, reason: "Сводка №55 недоступна" };
        }
      } else if (!(raw.reg55 && raw.reg55.available)) {
        raw.reg55 = { available: false, reason: "Сводка №55 недоступна" };
      }
      if (overviewResponse && overviewResponse.ok) {
        var ov = await overviewResponse.json();
        raw.attention = ov.attention || null;
        raw.zone_trends = ov.zone_trends || (ov.attention && ov.attention.zone_trends) || [];
        if (ov.by_doctor) raw.by_doctor = ov.by_doctor;
        if (ov.icd_visit_status) raw.icd_visit_status = ov.icd_visit_status;
        if (ov.clinical_gaps) raw.clinical_gaps = ov.clinical_gaps;
        if (ov.kp_unmatched) raw.kp_unmatched = ov.kp_unmatched;
        if (ov.worst_severity_cases) raw.worst_severity_cases = ov.worst_severity_cases;
        if (ov.severity_totals && !raw.severity_totals) raw.severity_totals = ov.severity_totals;
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
      var doctorId = row.doctor_id || "";
      if ((!doctor || doctor === "Врач не указан") && doctorId) doctor = "ID врача: " + doctorId;
      return { raw: row, id: id, visitId: row.visit_id || id, patientId: row.patient_id || "",
        doctorId: doctorId, date: row.date || row.visit_date || "", doctor: doctor, specialty: specialty,
        branch: row.filial || row.branch || "", diagnosis: diagnosis, total: total, status: status,
        kind: row.document_kind_label || row.kz_kind_label || row.kz_kind || "Не указан",
        coverage: firstNumeric([row.coverage_pct, row.coverage, row.deep_coverage_pct]),
        confidence: firstNumeric([row.confidence_pct, row.confidence, row.deep_confidence_pct]),
        reg55: firstNumeric([
          row.reg55_section_pct, row.reg55_pct, row.axis_regulatory, (row.axes || {}).regulatory
        ]),
        reg55Band: row.reg55_band || "",
        reg55Pack: row.reg55_pack || "",
        zone1Band: row.zone1_band || "", zone2aBand: row.zone2a_band || "",
        zone2bBand: row.zone2b_band || "", zone2bKp: row.zone2b_kp_status || "",
        attentionPrimary: row.attention_primary || "",
        attentionReason: row.attention_reason_ru || "" };
    }
    function reg55BandLabelRu(code) {
      return ({
        compliant_min: "Соответствует (мин. меры)",
        compliant_measures: "Соответствует (нужны меры)",
        noncompliant: "Не соответствует",
        unscored: "Не оценено"
      })[code] || code || "Не оценено";
    }
    function reg55BandPill(code, pct) {
      var c = String(code || "unscored");
      var tone = reg55BandTone(c);
      var label = reg55BandLabelRu(c);
      var pctBit = pct == null || pct === "" || !Number.isFinite(Number(pct)) ? "" :
        (' <b>' + Math.round(Number(pct)) + "%</b>");
      return '<span class="status ' + tone + ' reg55-band-pill" title="' + esc(label) + '">' +
        esc(label) + pctBit + "</span>";
    }
    function reg55RowClass(band) {
      if (band === "noncompliant") return " reg55-row--noncompliant";
      if (band === "compliant_measures") return " reg55-row--measures";
      return "";
    }
    function zoneBandChip(band, kpStatus) {
      var b = String(band || "na");
      if (b === "na" && kpStatus === "unmatched") {
        return '<span class="status muted">протокол не подобран</span>';
      }
      var map = {
        ok: ["good", "в норме"],
        weak: ["review", "слабо"],
        bad: ["critical", "плохо"],
        na: ["muted", "нет данных"]
      };
      var pair = map[b] || map.na;
      return '<span class="status ' + pair[0] + '">' + esc(pair[1]) + "</span>";
    }
    function layerLabelRu(primary) {
      return ({
        safety: "Риск", zone1: "Оформление", zone2a: "Диагноз", zone2b: "План по протоколу"
      })[primary] || "";
    }
    function findingZoneKey(finding) {
      var code = String((finding && (finding.code || finding.finding_code)) || "");
      if (/^C_/.test(code)) return "safety";
      if (/^B_complaint_exam|^B_dx_not|^B_tentative|^B_chronic_dx|^B_treatment_before|^finding_/i.test(code)) {
        return "zone2a";
      }
      if (/^B_dx|^B_icd|diagnosis/i.test(code)) return "zone2a";
      if (/plan|exam_rec|treat|follow|D_reg55|B_complaint_not_addressed/i.test(code)) return "zone2b";
      if (/^A_|missing|complain|anamnes|objective|mo_complete/i.test(code)) return "zone1";
      var axis = String((finding && finding.axis) || "");
      if (axis === "safety") return "safety";
      if (axis === "clinical_concordance") return "zone2a";
      if (axis === "regulatory") return "zone2b";
      if (axis === "documentation") return "zone1";
      return "other";
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
    function severityTone(item) {
      if (item && item.severity_tone) return String(item.severity_tone);
      var sev = String((item && item.severity) || "");
      if (sev === "P0") return "critical";
      if (sev === "P1") return "important";
      if (sev === "P2") return "check";
      if (sev === "P3") return "formal";
      return "review";
    }
    function severityLabel(item) {
      return (item && (item.severity_label_ru || item.priority_label_ru)) ||
        ({ P0: "Критично", P1: "Важно", P2: "Умеренно", P3: "Оформление" }[item && item.severity] ||
          (item && item.severity) || "Проверить");
    }
    function icdVisitChip(row) {
      var st = (row && (row.icd_visit_status || (row.raw && row.raw.icd_visit_status))) || "";
      if (!st || st === "unknown") return "";
      var label = (row && (row.icd_visit_status_label_ru || (row.raw && row.raw.icd_visit_status_label_ru))) || "";
      var title = (row && (row.icd_visit_status_title_ru || (row.raw && row.raw.icd_visit_status_title_ru))) || "";
      if (!label) {
        label = st === "ok" ? "МКБ ✓" : st === "missing_dx" ? "нет Dx" : st === "not_in_directory" ? "не в МКБ" : st === "weak_name" ? "слабо МКБ" : "";
      }
      if (!label) return "";
      var tone = st === "ok" ? "good" : (st === "weak_name" ? "review" : "critical");
      return ' <span class="status ' + tone + ' icd-visit-chip" title="' + esc(title || label) + '">' + esc(label) + "</span>";
    }
    function historyTierLabelRu(tier) {
      var map = {
        known_to_doctor: "код уже был у этого врача",
        known_in_specialty_only: "код был у коллег специальности",
        new_for_profile: "новый код для профиля у врача",
        first_contact: "первый контакт с этим врачом",
        insufficient: "истории недостаточно"
      };
      return map[String(tier || "")] || String(tier || "");
    }
    function historyVisitChip(row) {
      var raw = row && (row.raw || row) || {};
      var n = Number(raw.history_prior_n || 0);
      var tier = String(raw.history_tier || "");
      if (!n && !tier) return "";
      var label = n > 0 ? ("история: " + n) : "история: 0";
      if (tier === "first_contact") label = "первый к врачу";
      else if (tier === "known_to_doctor") label = "код уже был";
      else if (tier === "known_in_specialty_only") label = "код у коллег";
      else if (tier === "new_for_profile") label = "новый код";
      else if (tier === "insufficient") label = "нет истории";
      var tone = (tier === "first_contact" || tier === "new_for_profile" || tier === "insufficient") ? "review" : "good";
      var title = historyTierLabelRu(tier) || "История пациента до этого визита";
      return ' <span class="status ' + tone + ' history-visit-chip" title="' + esc(title) + '">' + esc(label) + "</span>";
    }
    function reg55BandTone(code) {
      if (code === "compliant_min") return "good";
      if (code === "compliant_measures") return "review";
      if (code === "noncompliant") return "critical";
      return "review";
    }
    function renderReg55(reg55, fallbackPct) {
      var pct = null;
      if (reg55 && reg55.reg55_section_pct != null && reg55.reg55_section_pct !== "") {
        pct = Number(reg55.reg55_section_pct);
      } else if (reg55 && reg55.regulatory_compliance_pct != null && reg55.regulatory_compliance_pct !== "") {
        pct = Number(reg55.regulatory_compliance_pct);
      } else if (fallbackPct != null && fallbackPct !== "") {
        pct = Number(fallbackPct);
      }
      if ((pct == null || !Number.isFinite(pct)) && !(reg55 && (reg55.criteria || []).length)) {
        return '<div class="detail-block reg55-block"><h3>Балл по постановлению МЗ №55</h3><p class="empty">' +
          esc((reg55 && reg55.note_ru) || "Нет данных для этого случая (оцениваются только клинические приёмы).") +
          "</p></div>";
      }
      var bandCode = (reg55 && reg55.reg55_band) || "";
      var bandLabel = (reg55 && reg55.reg55_band_label_ru) || "";
      var head = "Средний балл (п.12): <b style=\"font-size:1.35rem\">" +
        (pct == null || !Number.isFinite(pct) ? "-" : (Math.round(pct) + "%")) + "</b>";
      if (bandLabel) {
        head += ' <span class="status ' + reg55BandTone(bandCode) + '">' + esc(bandLabel) + "</span>";
      }
      if (reg55 && reg55.pack_label_ru) {
        head += "<br><small>" + esc(reg55.pack_label_ru) + "</small>";
      }
      if (reg55 && reg55.passed != null && reg55.total != null && reg55.total > 0) {
        head += "<br>полная оценка 1.0: " + esc(reg55.passed) + " из " + esc(reg55.total) + " применимых";
      }
      if (reg55 && reg55.na) head += " · n/a вне знаменателя: " + esc(reg55.na);
      var focusPoint = state.rubricCriterion || "";
      var criteria = (reg55 && reg55.criteria) || [];
      var rows = criteria.length ? criteria.map(function (item) {
        var verdict = item.verdict || "";
        var tone = verdict === "pass" ? "good" : (verdict === "fail" ? "critical" : "review");
        var scoreBit = item.score == null || verdict === "na" ? "n/a" : String(item.score);
        var point = item.point_no || item.point || "-";
        var wrong = item.whats_wrong_ru || (verdict === "fail" || verdict === "partial" ? (item.how_checked_ru || "") : "");
        var focus = focusPoint && String(focusPoint) === String(point) ? " rubric-row--focus" : "";
        var ev127 = item.evidence_from_127 ? ' <span class="card-sub">· опора №127</span>' : "";
        return '<tr class="' + focus + '">' +
          '<td><b>' + esc(point) + "</b></td>" +
          '<td>' + esc(item.title || item.id || "критерий") + ev127 +
          (item.group ? ('<br><small>' + esc(item.group) + "</small>") : "") +
          "</td>" +
          '<td><span class="status ' + tone + '">' + esc(item.verdict_ru || verdict || "-") +
          "</span> · " + esc(scoreBit) + "</td>" +
          '<td>' + (wrong ? esc(wrong) : (verdict === "na" ? "не учитывается в формуле" : " - ")) + "</td>" +
          "</tr>";
      }).join("") : "";
      var table = rows ?
        '<div class="table-wrap compact-table"><table><thead><tr><th>Пункт</th><th>Описание</th><th>Оценка</th><th>Что не так</th></tr></thead><tbody>' +
        rows + "</tbody></table></div>" :
        "<p class=\"card-sub\">Детализация пунктов появится после загрузки критериев №55.</p>";
      var formula = (reg55 && reg55.formula_ru) ||
        "Средний балл №55 = 100 × (сумма 0/0.5/1) / (применимые пункты разд. V; n/a вне знаменателя)";
      var measures = (reg55 && reg55.measures) || [];
      var measuresHtml = "";
      if (measures.length && (bandCode === "compliant_measures" || bandCode === "noncompliant")) {
        measuresHtml = "<h4 style=\"margin:12px 0 6px;font-size:13px\">Комплекс мероприятий (score &lt; 1)</h4><ul>" +
          measures.slice(0, 8).map(function (m) {
            return "<li><b>" + esc(m.point || "") + "</b> " + esc(m.title || "") +
              " - " + esc(m.reason || "") + "</li>";
          }).join("") + "</ul>";
      }
      return '<div class="detail-block reg55-block"><h3>Балл по постановлению МЗ №55</h3>' +
        '<p>' + head + "</p>" +
        '<p class="card-sub">' + esc(formula) + "</p>" +
        (reg55 && reg55.reg55_band_detail_ru ? ('<p class="card-sub">' + esc(reg55.reg55_band_detail_ru) + "</p>") : "") +
        (reg55 && reg55.note_ru ? ('<p class="card-sub">' + esc(reg55.note_ru) + "</p>") : "") +
        measuresHtml + table + "</div>";
    }
    function renderPatientHistory(bundle) {
      if (!bundle || !bundle.summary) {
        return '<div class="detail-block patient-history-block"><h3>История пациента</h3>' +
          '<p class="empty">Нет данных истории на складе (нет patient_id / patient_key или склад недоступен).</p>' +
          '<p class="card-sub">История нужна для контекста МКБ, подбора КП и динамики рубрики МЗ; ' +
          'по умолчанию не меняет итоговую оценку (shadow).</p></div>';
      }
      var summary = bundle.summary || {};
      var coverage = bundle.coverage || {};
      var tier = bundle.tier || "";
      var tierRu = bundle.tier_label_ru || historyTierLabelRu(tier) || tier;
      var nVisits = Number(summary.n_visits || 0);
      var head = "Всего " + nVisits + " визит(ов) до этого случая";
      if (coverage.first_date || coverage.last_date) {
        head += " · " + (coverage.first_date || "?") + " … " + (coverage.last_date || "?");
      }
      var usage = bundle.usage_for_scores_ru ||
        "История - контекст до визита. По умолчанию shadow (не двигает итоговую оценку). " +
        "Влияет на сверку названия МКБ, разрыв линии Dx, подбор КП, динамику рубрики МЗ и приоритет LLM-очереди.";
      function codesLine(title, codes) {
        codes = codes || {};
        var keys = Object.keys(codes);
        if (!keys.length) return "";
        return '<p class="card-sub">' + esc(title) + ": " +
          keys.slice(0, 8).map(function (code) {
            return esc(code) + "×" + esc(codes[code]);
          }).join(", ") + "</p>";
      }
      function shelfHtml(title, rows, collapsed) {
        rows = rows || [];
        if (!rows.length) {
          return "<details><summary>" + esc(title) + " (0)</summary>" +
            '<p class="empty">Нет визитов на этой полке.</p></details>';
        }
        var body = rows.slice(0, 12).map(function (visit) {
          var pct = visit.overall_pct == null ? "-" : (Math.round(Number(visit.overall_pct)) + "%");
          var mid = visit.visit_id || visit.mis_id || "";
          var kind = visit.document_kind ? (" · " + visit.document_kind) : "";
          return '<li><button type="button" class="linkish" data-case="' + esc(mid) + '">' +
            esc(visit.visit_date || "") + "</button> · " + esc(visit.diagnosis_code || "-") +
            (visit.diagnosis_text ? (" · " + esc(String(visit.diagnosis_text).slice(0, 60))) : "") +
            " · МО " + esc(pct) + esc(kind) + "</li>";
        }).join("");
        var open = collapsed ? "" : " open";
        return "<details" + open + "><summary>" + esc(title) + " (" + rows.length + ")</summary><ul class=\"history-visit-list\">" + body + "</ul></details>";
      }
      var emptyHint = nVisits === 0
        ? '<p class="empty">На складе нет более ранних визитов этого пациента (или это первый контакт). ' +
          'Это нормально для first_contact; не означает, что блок сломан.</p>'
        : "";
      return '<div class="detail-block patient-history-block"><h3>История пациента</h3>' +
        '<p><b>' + esc(tierRu || "контекст") + "</b></p>" +
        '<p class="card-sub">' + esc(head) + "</p>" +
        emptyHint +
        codesLine("Коды у этого врача", summary.codes_same_doctor) +
        codesLine("Коды у коллег специальности", summary.codes_same_specialty) +
        shelfHtml("К этому врачу", bundle.same_doctor, false) +
        shelfHtml("Другие врачи этой специальности", bundle.same_specialty, false) +
        shelfHtml("Прочие специальности", bundle.other, true) +
        '<details class="mo-secondary-details"><summary>Как история влияет на оценки</summary>' +
        '<p class="card-sub">' + esc(usage) + "</p></details></div>";
    }
    function documentRow(item) {
      var reason = item.attentionReason || "";
      var band = item.reg55Band || (item.raw && item.raw.reg55_band) || "";
      return '<tr tabindex="0" class="' + reg55RowClass(band).trim() + '" data-case="' + esc(item.id) + '"><td class="id-cell">' + esc(item.visitId || item.id || "-") +
        '</td><td class="id-cell">' + esc(item.patientId || "-") + '</td><td>' + esc(item.date) + '</td><td><b>' + esc(item.doctor) +
        '</b><br><small>' + esc(item.specialty) + '</small></td><td>' + esc(item.branch) + '</td><td>' + esc(item.diagnosis) +
        icdVisitChip(item.raw || item) + historyVisitChip(item.raw || item) +
        '</td><td>' + zoneBandChip(item.zone1Band) + '</td><td>' + zoneBandChip(item.zone2aBand) +
        '</td><td>' + zoneBandChip(item.zone2bBand, item.zone2bKp) +
        '</td><td>' + esc(reason || layerLabelRu(item.attentionPrimary) || "-") +
        '</td><td><span class="status ' + statusClass(item.status) + '">' +
        esc(statusLabel(item.status)) + "</span></td>" +
        '<td><b>' + esc(scoreLabel(item.total, item.raw.score_reason)) + '</b></td>' +
        '<td>' + reg55BandPill(band, item.reg55) + '</td>' +
        '<td>' + esc(score(item.coverage)) + '</td>' +
        '<td>' + esc(score(item.confidence)) + '</td></tr>';
    }
    function queueRow(item) {
      var raw = item.raw || {};
      var priority = raw.severity_label_ru || severityLabel(raw) ||
        (Number(raw.p0 || 0) > 0 ? "Критично" : Number(raw.p1 || 0) > 0 ? "Важно" : "Низкий балл");
      var tone = raw.severity_tone || severityTone(raw) || statusClass(item.status);
      var crm = raw.crm || {};
      var pdfUrl = raw.pdf_url || ("/api/methodist/mo/cases/" + encodeURIComponent(item.id) + "/pdf");
      var layer = raw.layer_ru || layerLabelRu(item.attentionPrimary || raw.attention_primary);
      var reason = item.attentionReason || raw.attention_reason_ru || raw.reason || raw.comment || "Требует ручной проверки";
      var band = item.reg55Band || raw.reg55_band || "";
      if (band && band !== "compliant_min" && reason.indexOf("№55") < 0) {
        reason = "№55 " + reg55BandLabelRu(band) + (reason ? " · " + reason : "");
      }
      var shadow = raw.shadow_dx_plan || item.shadow_dx_plan || {};
      var shadowBand = shadow.case_attention_band || "";
      var shadowBadge = (shadowBand === "poor" || shadowBand === "critical")
        ? ('<br><span class="' + shadowBandClass(shadowBand) + '" title="' +
          esc(shadow.disclaimer_ru || "shadow") + '">' + esc(shadowBandLabel(shadowBand)) + "</span>")
        : "";
      return '<tr tabindex="0" class="' + reg55RowClass(band).trim() + '" data-case="' + esc(item.id) + '"><td><input type="checkbox" data-case-select="' + esc(item.id) + '" aria-label="Выбрать случай"></td><td><span class="status ' +
        esc(tone) + '">' + esc(priority) + '</span>' + shadowBadge + '</td><td>' + esc(layer || "-") +
        '</td><td class="id-cell">' + esc(item.visitId || item.id || "-") +
        '</td><td class="id-cell">' + esc(item.patientId || "-") + '</td><td>' + esc(item.date) +
        '</td><td>' + esc(item.branch) + '</td><td><b>' + esc(item.doctor) + '</b><br><small>' + esc(item.specialty) +
        '</small></td><td>' + esc(item.diagnosis) + icdVisitChip(item.raw || item) + historyVisitChip(item.raw || item) +
        '</td><td>' + zoneBandChip(item.zone1Band || raw.zone1_band) +
        '</td><td>' + zoneBandChip(item.zone2aBand || raw.zone2a_band) +
        '</td><td>' + zoneBandChip(item.zone2bBand || raw.zone2b_band, item.zone2bKp || raw.zone2b_kp_status) +
        '</td><td>' + reg55BandPill(band, item.reg55) +
        '</td><td>' + esc(reason) + '</td><td>' +
        esc(raw.assignee || crm.assignee || "Не назначен") + '</td><td>' + esc(raw.due_date || crm.due_date || "Сегодня") +
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
    function isSingleDayPeriod() {
      return !!(state.dateFrom && state.dateTo && state.dateFrom === state.dateTo);
    }
    function bindSortableHeaders(table) {
      if (!table) return;
      var body = table.tBodies && table.tBodies[0];
      attachTableChrome(table, {
        id: body && body.id ? ("chrome-" + body.id) : "chrome-cases",
        serverSort: true
      });
    }
    async function loadCases(queue) {
      var q = query();
      q.set("page", state.pageNo);
      q.set("page_size", isSingleDayPeriod() ? "100" : "50");
      if (queue) q.set("queue_only", "1");
      var response = await request("/cases?" + q.toString(), "/cases?" + q.toString());
      if (!response.ok) throw new Error("Не удалось загрузить случаи.");
      var data = await response.json();
      var rows = (data.rows || data.cases || data.items || data.worst_visits || []).map(rowRecord);
      state.caseNavIds = rows.map(function (item) { return item.id; }).filter(Boolean);
      var body = queue ? $("queue-rows") : $("document-rows");
      var emptyState = data.empty_state || {};
      var pageHost = queue ? $("page-queue") : $("page-documents");
      var banner = pageHost ? pageHost.querySelector(".day-table-banner") : null;
      if (!queue && pageHost) {
        if (!banner) {
          banner = document.createElement("div");
          banner.className = "day-table-banner";
          var card = pageHost.querySelector(".card");
          if (card) pageHost.insertBefore(banner, card);
        }
        banner.hidden = false;
        var scopeNote = "Показаны только клинические приёмы. Процедуры, профосмотры, диагностика и стоматология в таблицу не входят и не оцениваются.";
        if (isSingleDayPeriod()) {
          banner.innerHTML = "<b>Таблица за " + esc(state.dateFrom) + "</b> · всего " +
            esc(data.total || rows.length) + " записей. " + esc(scopeNote) +
            " Колонка «Балл №55» - средний % по формуле пост. МЗ №55.";
        } else {
          banner.innerHTML = "<b>Все случаи</b> · " + esc(data.total || rows.length) + " записей. " + esc(scopeNote);
        }
      }
      body.innerHTML = rows.length ? rows.map(queue ? queueRow : documentRow).join("") :
        '<tr><td colspan="' + (queue ? 14 : 12) + '" class="empty"><b>' +
        esc(emptyState.title || "По выбранным фильтрам случаев нет.") + "</b><div>" +
        esc(emptyState.hint || "Измените фильтры или расширьте период.") + "</div></td></tr>";
      bindCaseRows(body);
      applyColumnVisibility(queue ? "queue" : "documents");
      bindSortableHeaders(body.closest("table"));
      if (!queue) {
        var pageSize = Number(data.page_size || (isSingleDayPeriod() ? 100 : 50));
        var total = Number(data.total || rows.length), pages = Math.max(1, Math.ceil(total / pageSize));
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
    function verdictTone(verdict) {
      var value = String(verdict || "").toLowerCase();
      if (/critical|poor/.test(value)) return "critical";
      if (/good|acceptable/.test(value)) return "good";
      return "review";
    }
    function judgeKpiCard(label, kpiData) {
      var payload = kpiData || {};
      var pct = payload.score_pct;
      var verdict = payload.verdict || "";
      var tone = verdictTone(verdict);
      return '<article class="kpi llm-judge-kpi llm-judge-kpi--' + tone + '">' +
        '<div class="kpi-label">' + esc(label) + '</div>' +
        '<div class="kpi-value">' + esc(score(pct)) + '</div>' +
        '<div class="kpi-meta"><span class="status ' + tone + '">' +
        esc(statusLabel(verdict) || "Нет вердикта") + '</span></div>' +
        (payload.summary_ru ? '<p class="llm-judge-kpi-summary">' + esc(payload.summary_ru) + '</p>' : "") +
        '</article>';
    }
    function judgeBlockChip(name, block) {
      var present = !!(block && block.present);
      var adequate = !!(block && block.adequate);
      var tone = !present ? "critical" : (adequate ? "good" : "review");
      var label = present ? (adequate ? "есть" : "слабо") : "пусто";
      return '<li class="llm-judge-block-chip llm-judge-block-chip--' + tone + '"><b>' + esc(name) +
        '</b><span class="status ' + tone + '">' + esc(label) + '</span>' +
        (block && block.note ? '<small>' + esc(block.note) + '</small>' : "") + '</li>';
    }
    function judgeList(items, emptyText) {
      if (!(items || []).length) return '<p class="empty">' + esc(emptyText || "Нет данных") + '</p>';
      return '<ul class="llm-judge-bullets">' + items.map(function (item) {
        return '<li>' + esc(item) + '</li>';
      }).join("") + '</ul>';
    }
    function renderLlmActionJudge(judge, documentData, item) {
      var clinical = (documentData && documentData.clinical) || {};
      var patient = (judge && judge.patient) || {};
      var metaLine = [
        item.doctor, item.specialty, item.date,
        patient.age_years != null ? ("возраст " + patient.age_years) : "",
        patient.audience && patient.audience !== "unknown" ? patient.audience : ""
      ].filter(Boolean).join(" · ");
      var head = '<div class="detail-block llm-judge-block"><h3>Разбор модели: три вопроса <span class="status review finding-shadow-badge">черновик</span></h3>' +
        '<p class="card-sub">' + esc(metaLine || "Врач и дата не указаны") +
        ' · полнота · диагноз · рекомендации</p>';
      if (!judge || !judge.available) {
        return head + '<p class="empty">LLM-оценка action-очереди ещё не готова. ' +
          esc((judge && judge.reason) || "После прогона batch появится здесь.") +
          '</p></div>';
      }
      var kpis = judge.kpis || {};
      var conclusions = judge.conclusions || {};
      var detail = judge.detail || {};
      var completeness = detail.completeness || {};
      var diagnosis = detail.diagnosis || {};
      var recommendations = detail.recommendations || {};
      var blockLabels = {
        complaints: "Жалобы", anamnesis: "Анамнез", objective_status: "Статус",
        exam_data: "Обследования", diagnosis: "Диагноз",
        exam_recommendations: "Рек. обслед.", treatment_recommendations: "Рек. леч."
      };
      var blocks = completeness.blocks || {};
      var blocksHtml = '<ul class="llm-judge-blocks">' + Object.keys(blockLabels).map(function (key) {
        return judgeBlockChip(blockLabels[key], blocks[key]);
      }).join("") + '</ul>';
      var slotPairs = [
        ["complaints", "Жалобы"], ["anamnesis_doctor", "Анамнез"], ["anamnesis_auto", "Анамнез (авто)"],
        ["objective_status", "Объективный статус"], ["exam_data", "Обследования"],
        ["clinical_diagnosis", "Диагноз"], ["exam_recommendations", "Рек. по обследованию"],
        ["treatment_recommendations", "Рек. по лечению"]
      ];
      var moSlots = slotPairs.filter(function (pair) { return clinical[pair[0]]; }).map(function (pair) {
        return '<section class="llm-judge-slot"><h4>' + esc(pair[1]) + '</h4><p>' +
          esc(String(clinical[pair[0]]).slice(0, 520)) +
          (String(clinical[pair[0]]).length > 520 ? "…" : "") + '</p></section>';
      }).join("");
      var icd = diagnosis.icd || {};
      var dxWhy = '<section class="llm-judge-slot"><h4>Диагноз</h4><p>' +
        esc(diagnosis.summary_ru || conclusions.diagnosis_ru || "Нет вывода") + '</p>' +
        (icd.code || icd.text ? '<p class="card-sub">МКБ: ' + esc([icd.code, icd.text, icd.fit].filter(Boolean).join(" · ")) + '</p>' : "") +
        (diagnosis.blocked_by_incomplete ? '<p class="inline-note">Оценка Dx ограничена пустыми блоками</p>' : "") +
        '<div class="llm-judge-split"><div><h5>Подтверждает</h5>' +
        judgeList(diagnosis.supported_by, "Нет опор") + '</div><div><h5>Не закрыто диагнозом</h5>' +
        judgeList(diagnosis.not_supported_by, "Пробелов не отмечено") + '</div></div></section>';
      var outcomeRu = { ok: "ок", contradiction: "противоречие", gap: "пробел", missing: "нет связи" };
      var chainHtml = (diagnosis.chain || []).length ? '<section class="llm-judge-slot"><h4>Цепочка клиника → диагноз</h4>' +
        (diagnosis.chain || []).map(function (link) {
          return '<div class="llm-judge-chain"><span class="status ' +
            (link.outcome === "ok" ? "good" : (link.outcome === "contradiction" ? "critical" : "review")) +
            '">' + esc(outcomeRu[link.outcome] || link.outcome || "?") + '</span> ' +
            esc((link.from || "") + " → " + (link.to || "")) +
            (link.note ? '<small>' + esc(link.note) + '</small>' : "") + '</div>';
        }).join("") + '</section>' : "";
      var exam = recommendations.exam || {};
      var treatment = recommendations.treatment || {};
      var follow = recommendations.follow_up || {};
      var planWhy =
        '<section class="llm-judge-slot"><h4>Обследование ' + esc(score(exam.score_pct)) +
        ' · ' + esc(statusLabel(exam.verdict || "")) + '</h4><p>' + esc(exam.summary_ru || "-") + '</p>' +
        '<div class="llm-judge-split"><div><h5>Назначено</h5>' + judgeList(exam.present, "Пусто") +
        '</div><div><h5>Не хватает</h5>' + judgeList(exam.missing_suggested, "Не указано") +
        '</div></div></section>' +
        '<section class="llm-judge-slot"><h4>Лечение ' + esc(score(treatment.score_pct)) +
        ' · ' + esc(statusLabel(treatment.verdict || "")) + '</h4><p>' + esc(treatment.summary_ru || "-") + '</p>' +
        '<div class="llm-judge-split"><div><h5>Назначения</h5>' + judgeList(treatment.present, "Пусто") +
        '</div><div><h5>Риски</h5>' + judgeList(treatment.concerns, "Без замечаний") +
        '</div></div></section>' +
        '<section class="llm-judge-slot"><h4>Наблюдение / явка ' + esc(score(follow.score_pct)) +
        ' · ' + esc(statusLabel(follow.verdict || "") || follow.kind || "") + '</h4><p>' +
        esc(follow.summary_ru || "-") + '</p></section>';
      var findingsHtml = (conclusions.findings || []).slice(0, 10).map(function (finding) {
        var stageRu = finding.stage === "a" || finding.stage === "A" ? "этап A (диагноз)" :
          (finding.stage === "b" || finding.stage === "B" ? "этап B (план)" : ("этап " + (finding.stage || "?")));
        return '<div class="llm-judge-finding"><span class="status ' +
          esc(severityTone(finding)) + '">' + esc(severityLabel(finding)) +
          '</span> <span class="card-sub">' + esc(stageRu) + '</span> ' +
          esc(finding.text_ru || finding.code || "") +
          (finding.evidence ? '<blockquote>«' + esc(finding.evidence) + '»</blockquote>' : "") +
          '</div>';
      }).join("");
      var queueNote = judge.queue_reason ?
        '<p class="inline-note">Почему в очереди: ' + esc(judge.queue_severity || "") +
        (judge.queue_severity ? " · " : "") + esc(judge.queue_reason) + '</p>' : "";
      var footerMeta = [
        conclusions.needs_human ? "нужен разбор методиста" : "можно закрыть после проверки",
        conclusions.confidence_a != null ? ("уверенность A " + Math.round(Number(conclusions.confidence_a) * 100) + "%") : "",
        conclusions.confidence_b != null ? ("уверенность B " + Math.round(Number(conclusions.confidence_b) * 100) + "%") : ""
      ].filter(Boolean).join(" · ");
      return head + queueNote +
        '<div class="drawer-grid llm-judge-kpis">' +
        judgeKpiCard("1. Полнота", kpis.completeness) +
        judgeKpiCard("2. Диагноз", kpis.diagnosis) +
        judgeKpiCard("3. Рекомендации", kpis.recommendations) +
        '</div>' +
        '<div class="llm-judge-section"><h4>Полнота блоков МО</h4>' + blocksHtml +
        (conclusions.completeness_ru ? '<p>' + esc(conclusions.completeness_ru) + '</p>' : "") + '</div>' +
        '<div class="llm-judge-compare">' +
        '<div><h4>Реальное МО</h4>' + (moSlots || '<p class="empty">Клинический текст недоступен</p>') + '</div>' +
        '<div><h4>Разбор модели</h4>' + dxWhy + chainHtml + planWhy +
        (findingsHtml ? '<div class="llm-judge-findings"><h4>Замечания модели</h4>' + findingsHtml + '</div>' : "") +
        ((conclusions.stage_a_ru || conclusions.stage_b_ru) ?
          '<section class="llm-judge-slot"><h4>Итог этапов</h4>' +
          (conclusions.stage_a_ru ? '<p><b>A (диагноз):</b> ' + esc(conclusions.stage_a_ru) + '</p>' : "") +
          (conclusions.stage_b_ru ? '<p><b>B (план):</b> ' + esc(conclusions.stage_b_ru) + '</p>' : "") +
          '</section>' : "") +
        '</div></div>' +
        '<p class="card-sub">Черновик модели · не меняет итоговый балл витрины · ' + esc(footerMeta) + '</p></div>';
    }
    function verdictLabelRu(value) {
      return ({ unreviewed: "не проверено", agree: "согласен", partial: "частично", disagree: "не согласен" })[value] || value || "-";
    }
    function protocolViewerUrl(item) {
      if (!item) return "";
      var viewer = item.viewer_url || (item.source_path ? ("/proto-viewer.html?path=" + encodeURIComponent(item.source_path)) : "");
      if (viewer.indexOf("/proto?") === 0) {
        viewer = "/proto-viewer.html?" + viewer.slice("/proto?".length);
      }
      if (viewer && item.section && viewer.indexOf("section=") < 0) {
        viewer += (viewer.indexOf("?") >= 0 ? "&" : "?") + "section=" + encodeURIComponent(item.section);
      }
      var anchorPage = item.page || item.anchor_page || item.page_start;
      if (viewer && anchorPage && viewer.indexOf("page=") < 0) {
        viewer += (viewer.indexOf("?") >= 0 ? "&" : "?") + "page=" + encodeURIComponent(String(anchorPage));
      }
      if (viewer && viewer.indexOf("from=") < 0) {
        viewer += (viewer.indexOf("?") >= 0 ? "&" : "?") + "from=mo";
      }
      return viewer;
    }
    function bindProtocolSuggestHost(host) {
      if (!host) return;
      var expand = host.querySelector("#protocol-suggest-expand");
      if (expand) {
        expand.addEventListener("click", function () {
          host.querySelectorAll("[data-protocol-extra]").forEach(function (node) {
            node.hidden = false;
          });
          expand.hidden = true;
        });
      }
      host.querySelectorAll("[data-retry-protocol-suggest]").forEach(function (button) {
        button.addEventListener("click", function () {
          if (state.openCaseId) loadProtocolSuggestIntoCase(state.openCaseId);
        });
      });
    }
    function renderProtocolSuggest(suggest) {
      state.protocolSuggest = suggest || null;
      if (!suggest || !suggest.available) {
        return '<div class="detail-block protocol-suggest-block"><h3>Протоколы МЗ</h3>' +
          '<p class="empty">' + esc((suggest && suggest.reason) || "Нет клинического протокола МЗ по этому диагнозу") +
          '</p><p class="card-sub">Без подобранного протокола план не штрафуем за несоответствие протоколу.</p>' +
          '<button type="button" class="button secondary compact" data-retry-protocol-suggest>Повторить подбор</button></div>';
      }
      var list = suggest.items || [];
      if (!list.length) {
        return '<div class="detail-block protocol-suggest-block"><h3>Протоколы МЗ</h3>' +
          '<p class="empty">' + esc(suggest.reason || "Нет клинического протокола МЗ по этому диагнозу") +
          '</p><p class="card-sub">Протокол не подобран - план не штрафуем за несоответствие протоколу.</p></div>';
      }
      var top = list[0];
      var topViewer = protocolViewerUrl(top);
      var topSearchQ = top.search_query || suggest.search_query || "";
      var topSearch = top.search_url || suggest.search_url ||
        (topSearchQ ? ("/doctor/search?q=" + encodeURIComponent(topSearchQ)) : "");
      var topBar = '<div class="protocol-suggest-top"><span>Протокол:</span><b>' +
        esc(top.title || "без названия") + '</b>' +
        (topViewer ? '<a class="button compact" href="' + esc(topViewer) + '" target="_blank" rel="noopener">Открыть протокол</a>' : "") +
        (topSearch ? '<a class="button secondary compact" href="' + esc(topSearch) + '" target="_blank" rel="noopener">Поиск в каталоге</a>' : "") +
        (list.length > 1 ? '<button type="button" class="linkish" id="protocol-suggest-expand">ещё ' +
          (list.length - 1) + '</button>' : "") +
        '</div>';
      var items = list.map(function (item, index) {
        var pid = item.protocol_id || ("idx-" + index);
        var reasons = (item.reasons || []).slice(0, 3).map(function (reason) {
          return '<li>' + esc(reason.text || reason.code || "") + '</li>';
        }).join("");
        var viewer = protocolViewerUrl(item);
        var searchQ = item.search_query || suggest.search_query || "";
        var searchUrl = item.search_url || suggest.search_url ||
          (searchQ ? ("/doctor/search?q=" + encodeURIComponent(searchQ)) : "");
        var titleHtml = viewer
          ? ('<a class="protocol-suggest-title-link" href="' + esc(viewer) + '" target="_blank" rel="noopener">' +
            esc(item.title || "Протокол") + '</a>')
          : esc(item.title || "Протокол");
        return '<article class="protocol-suggest-item" data-protocol-id="' + esc(pid) + '"' +
          (index > 0 ? ' hidden data-protocol-extra="1"' : "") + '>' +
          '<div class="protocol-suggest-title"><b>' + (index + 1) + ". " + titleHtml + '</b></div>' +
          '<div class="protocol-suggest-meta"><span class="status review">' +
          esc(item.match_kind_label || item.match_kind || "клиника") + '</span><span>' +
          esc(item.score != null ? (Math.round(Number(item.score)) + " баллов") : "") +
          '</span>' +
          (viewer ? '<a class="button compact" href="' + esc(viewer) +
            '" target="_blank" rel="noopener">Открыть протокол</a>' : "") +
          (searchUrl ? '<a class="button secondary compact" href="' + esc(searchUrl) +
            '" target="_blank" rel="noopener">Поиск в каталоге</a>' : "") +
          '</div>' + (reasons ? '<ul class="llm-judge-bullets">' + reasons + '</ul>' : "") +
          '<div class="protocol-suggest-rates" role="radiogroup" aria-label="Релевантность протокола">' +
          [['relevant','да'],['partial','частично'],['irrelevant','нет'],['unreviewed','не оценил']].map(function (pair) {
            return '<label><input type="radio" name="proto-rate-' + esc(pid) + '" value="' + pair[0] + '"' +
              (pair[0] === "unreviewed" ? " checked" : "") + '> ' + pair[1] + '</label>';
          }).join("") + '</div></article>';
      }).join("");
      return '<div class="detail-block protocol-suggest-block"><h3>Протоколы МЗ</h3>' +
        '<p class="card-sub">«Открыть протокол» - навигация и PDF по страницам; «Поиск в каталоге» - если нужен другой КП. Без КП план не штрафуем.</p>' +
        topBar + items + '</div>';
    }
    function verdictSelect(id, current) {
      var options = [
        ["unreviewed", "Не проверено"],
        ["agree", "Ок"],
        ["partial", "Замечание"],
        ["disagree", "Не применимо"]
      ];
      return '<select class="control" id="' + id + '">' + options.map(function (option) {
        return '<option value="' + option[0] + '"' + (option[0] === (current || "unreviewed") ? " selected" : "") + ">" +
          option[1] + "</option>";
      }).join("") + "</select>";
    }
    function renderReviewPackHistory(packs) {
      if (!(packs || []).length) {
        return '<div class="detail-block review-pack-history"><h3>История разборов</h3><p class="empty">Сохранённых пакетов пока нет.</p></div>';
      }
      return '<div class="detail-block review-pack-history"><h3>История разборов</h3>' + packs.map(function (pack, index) {
        var summary = pack.decision_summary || {};
        return '<article class="review-pack-item"><div><b>v' + (packs.length - index) + '</b> · ' +
          esc(new Date(pack.created_at).toLocaleString("ru-RU")) + ' · ' + esc(pack.actor || "методист") +
          (pack.training_use ? ' · <span class="status good">для обучения</span>' : "") +
          '</div><div class="card-sub">' +
          esc(["Полнота: " + verdictLabelRu(summary.verdict_completeness),
            "Диагноз: " + verdictLabelRu(summary.verdict_diagnosis),
            "Рек.: " + verdictLabelRu(summary.verdict_recommendations)].join(" · ")) +
          '</div>' + (summary.summary_ru ? '<p>' + esc(summary.summary_ru) + '</p>' : "") +
          '<div class="row-actions"><button class="button secondary compact" type="button" data-load-pack="' +
          esc(pack.pack_id) + '">Открыть / исправить</button></div></article>';
      }).join("") + '</div>';
    }
    function shadowBandLabel(band) {
      if (band === "critical") return "Критично (shadow)";
      if (band === "poor") return "Плохо (shadow)";
      return "Без красного флага";
    }
    function shadowBandClass(band) {
      if (band === "critical") return "status bad";
      if (band === "poor") return "status warn";
      return "status good";
    }
    function renderShadowDxPlan(shadow) {
      shadow = shadow || {};
      var disclaimer = shadow.disclaimer_ru || "Клиническая калибровка (shadow) - не официальная оценка";
      if (!shadow.available) {
        return '<div class="detail-block mo-shadow-dx-plan">' +
          "<h3>" + esc(disclaimer) + "</h3>" +
          '<p class="card-sub">' + esc(shadow.reason || "Ещё не посчитано для этого случая") + "</p></div>";
      }
      function line(title, block) {
        block = block || {};
        var att = block.attention || {};
        var band = att.band || "none";
        var score = block.score_pct != null ? Math.round(Number(block.score_pct)) + "%" : "-";
        var summary = block.summary_ru || "";
        var ens = block.ensemble_pct != null ? (" · ensemble " + Math.round(Number(block.ensemble_pct)) + "%") : "";
        return '<div class="mo-shadow-endpoint"><b>' + esc(title) + "</b> · " +
          '<span class="' + shadowBandClass(band) + '">' + esc(shadowBandLabel(band)) + "</span> · " +
          esc(score) + esc(ens) +
          (summary ? '<p class="card-sub">' + esc(summary) + "</p>" : "") +
          "</div>";
      }
      return '<div class="detail-block mo-shadow-dx-plan">' +
        "<h3>" + esc(disclaimer) + "</h3>" +
        '<p class="card-sub">Красное только при poor/critical после смягчения порогов. Official scores не меняются.</p>' +
        line("Диагноз", shadow.dx) +
        line("План", shadow.plan) +
        "</div>";
    }
    function renderZonesHero(zones) {
      if (!zones || !zones.ok || zones.skipped) return "";
      var zoneMap = { zone1: "documentation", zone2a: "diagnosis", zone2b: "plan" };
      var cards = [
        ["zone1", "Оформление"],
        ["zone2a", "Диагноз"],
        ["zone2b", "План по протоколу"]
      ].map(function (pair) {
        var z = zones[pair[0]] || {};
        var why = "";
        (zones.criteria || []).some(function (c) {
          if (String(c.zone || "") === zoneMap[pair[0]] && (c.score === 0 || c.score === 0.5 || c.na_reason)) {
            why = c.reason || "";
            return true;
          }
          return false;
        });
        return '<article class="zone-card zone-card--' + esc(z.band || "na") + '" data-zone-filter="' + pair[0] + '">' +
          '<div class="zone-card-label">' + esc(z.label_ru || pair[1]) + '</div>' +
          '<div class="zone-card-band">' + zoneBandChip(z.band, z.kp_status) + '</div>' +
          (why ? '<p class="zone-card-why">' + esc(String(why).slice(0, 140)) + '</p>' : "") +
          '</article>';
      }).join("");
      var safety = (zones.safety || {}).band;
      var risk = safety && safety !== "none" ? '<span class="status critical zone-risk-badge">Риск</span>' : "";
      return '<div class="zones-hero"><div class="zones-hero-head"><h3>Оценка случая</h3>' + risk +
        '</div><div class="zones-hero-grid">' + cards + '</div></div>';
    }
    function renderFindingsCompact(findings, crm, llmJudge) {
      var filters = [
        ["all", "Все"], ["zone1", "Оформление"], ["zone2a", "Диагноз"],
        ["zone2b", "План"], ["safety", "Риск"]
      ];
      var chips = '<div class="zone-finding-filters">' + filters.map(function (pair, idx) {
        return '<button type="button" class="button secondary compact' + (idx === 0 ? " is-active" : "") +
          '" data-finding-zone="' + pair[0] + '">' + esc(pair[1]) + '</button>';
      }).join("") + '</div>';
      var list = findings.length ? findings.map(function (finding) {
        var zkey = findingZoneKey(finding);
        var title = finding.title_ru || finding.title || finding.code || "Замечание";
        var decision = (crm.finding_decisions || {})[finding.code] || "unreviewed";
        var linked = finding.linked_fields || [];
        return '<article class="finding-card finding-card--compact" data-finding-zone-item="' + zkey + '">' +
          '<div class="finding-card-head"><span class="status muted">' + esc(layerLabelRu(zkey) || "Прочее") +
          '</span><span class="status ' + esc(finding.severity_tone || severityTone(finding)) + '">' +
          esc(finding.severity_label_ru || severityLabel(finding) || "Проверить") + '</span></div>' +
          '<div class="finding-card-title">' + esc(title) + '</div>' +
          (finding.detail_ru || finding.detail ? '<p class="finding-detail">' +
            esc(String(finding.detail_ru || finding.detail).slice(0, 220)) + '</p>' : "") +
          (linked[0] ? '<button type="button" class="linkish" data-focus-clinical="' + esc(linked[0]) +
            '">показать в тексте МО</button>' : "") +
          (finding.code ? '<label class="filter finding-decision"><span>Решение</span><select class="control" data-finding-code="' +
            esc(finding.code) + '"><option value="unreviewed"' + (decision === "unreviewed" ? " selected" : "") +
            '>Не проверено</option><option value="confirmed"' + (decision === "confirmed" ? " selected" : "") +
            '>Подтверждено</option><option value="false_positive"' + (decision === "false_positive" ? " selected" : "") +
            '>Отклонено</option></select></label>' : "") +
          '</article>';
      }).join("") : '<p class="empty">Замечаний нет.</p>';
      var llmLine = "";
      if (llmJudge && llmJudge.available) {
        var k = llmJudge.kpis || {};
        llmLine = '<p class="card-sub llm-inline">ИИ: оформление - ' +
          esc(statusLabel((k.completeness || {}).verdict) || "нет") +
          '; диагноз - ' + esc(statusLabel((k.diagnosis || {}).verdict) || "нет") +
          '; план - ' + esc(statusLabel((k.recommendations || {}).verdict) || "нет") + '</p>';
      }
      return '<div class="detail-block"><h3>Что не так</h3>' + chips + llmLine +
        '<div class="findings-compact-list">' + list + '</div></div>';
    }
    function renderHistoryContinuity(cont) {
      if (!cont || !cont.mode) return "";
      var track = cont.deep_run_track || "";
      var tone = track === "safety" || track === "history" ? "review" : (track === "strong_model" ? "review" : "good");
      var already = (cont.already_described || []).length
        ? "Уже встречалось: " + (cont.already_described.indexOf("diagnosis") >= 0 ? "диагноз" : "описание") +
          (cont.last_matched_date ? (" (" + cont.last_matched_date + ")") : "")
        : "";
      return '<p><b>' + esc(cont.mode_ru || "") + "</b></p>" +
        (already ? '<p class="card-sub">' + esc(already) + "</p>" : "") +
        '<p class="card-sub"><span class="status ' + tone + '">' + esc(cont.deep_run_track_ru || "") +
        "</span></p>" +
        (cont.usage_ru ? '<p class="card-sub">' + esc(cont.usage_ru) + "</p>" : "");
    }
    function renderHistoryCompact(bundle) {
      if (!bundle || !bundle.summary) {
        return '<div class="detail-block patient-history-block"><h3>История пациента</h3>' +
          '<p class="empty">Нет prior - коррекции плана не оцениваются.</p></div>';
      }
      var summary = bundle.summary || {};
      var n = Number(summary.n_visits || 0);
      var sameDoc = (bundle.same_doctor || []).length;
      var sameSpec = (bundle.same_specialty || []).length;
      var prior = n > 0 ? "есть prior" : "нет prior";
      var deep = bundle.deep || {};
      var deepSlots = (deep.already_slots || []).join(", ");
      var deepLine = deep.prior_visit_date
        ? ('<p class="card-sub">Слоты прошлого визита ' + esc(deep.prior_visit_date) +
          (deepSlots ? (": " + esc(deepSlots)) : "") + "</p>")
        : "";
      return '<div class="detail-block patient-history-block"><h3>История пациента</h3>' +
        renderHistoryContinuity(bundle.continuity) +
        deepLine +
        '<p>К этому врачу: ' + sameDoc + ' · К специальности: ' + sameSpec +
        ' · Всего: ' + n + ' · Для коррекций плана: ' + prior + '</p>' +
        (n === 0 ? '<p class="card-sub">Коррекции плана не оцениваются, если на складе нет более ранних визитов с ключом пациента.</p>' : "") +
        '<details><summary>Показать визиты</summary>' + renderPatientHistory(bundle) + '</details></div>';
    }
    function renderReviewBrief(brief, narrative) {
      if (!brief || !brief.available || !brief.ok) {
        return '<div class="detail-block review-brief-block"><h3>Итог разбора</h3>' +
          '<p class="empty">' + esc((brief && brief.reason) || "Итог разбора пока недоступен.") +
          '</p></div>';
      }
      var zones = brief.zones || {};
      var axes = brief.diagnosis_axes || {};
      var weak = (brief.methodology_weak || []).slice(0, 6).map(function (item) {
        var mark = item.score === 0 ? "плохо" : (item.score === 0.5 ? "слабо" : "н/д");
        return '<li><b>' + esc(item.title || item.id || "") + '</b> (' + esc(mark) + ')' +
          (item.reason ? ' - ' + esc(item.reason) : "") + '</li>';
      }).join("");
      var gaps = (brief.clinical_gaps || []).slice(0, 6).map(function (item) {
        return '<li><b>' + esc(item.title_ru || item.code || "") + '</b>' +
          (item.detail_ru ? ' - ' + esc(String(item.detail_ru).slice(0, 180)) : "") + '</li>';
      }).join("");
      var feedback = (brief.doctor_feedback || []).map(function (line) {
        return '<li>' + esc(line) + '</li>';
      }).join("");
      var icd = brief.icd || {};
      var proto = brief.protocol || {};
      var method = (axes.methodology || {});
      var icdAxis = (axes.icd_directory || {});
      var support = (axes.clinical_support || {});
      var ai = "";
      if (narrative && narrative.available) {
        ai = '<div class="review-brief-ai"><h4>Черновик ИИ</h4><p>' +
          esc(narrative.summary_ru || "") + '</p>' +
          ((narrative.doctor_feedback_ru || []).length ?
            '<ul>' + narrative.doctor_feedback_ru.slice(0, 4).map(function (line) {
              return '<li>' + esc(line) + '</li>';
            }).join("") + '</ul>' : "") +
          '<p class="card-sub">Не меняет зоны методики · уверенность ' +
          esc(narrative.confidence != null ? Math.round(Number(narrative.confidence) * 100) + "%" : " - ") +
          '</p></div>';
      }
      return '<div class="detail-block review-brief-block"><h3>Итог разбора</h3>' +
        '<p class="card-sub">' + esc(brief.summary_ru || "") + '</p>' +
        '<div class="review-brief-zones">' +
        '<div><b>Оформление</b> - ' + esc((zones.documentation || {}).band_ru || " - ") +
        '<div class="card-sub">' + esc((zones.documentation || {}).why_ru || "") + '</div></div>' +
        '<div><b>Диагноз</b> - ' + esc((zones.diagnosis || {}).band_ru || " - ") +
        '<div class="card-sub">' + esc((zones.diagnosis || {}).why_ru || "") + '</div></div>' +
        '<div><b>План</b> - ' + esc((zones.plan || {}).band_ru || " - ") +
        '<div class="card-sub">' + esc((zones.plan || {}).why_ru || "") + '</div></div>' +
        '</div>' +
        '<h4>Три оси диагноза</h4>' +
        '<ul class="review-brief-axes">' +
        '<li><b>' + esc(method.label_ru || "Методика") + '</b>: ' +
        esc(method.band_ru || "") + ' - ' + esc(method.detail_ru || "") + '</li>' +
        '<li><b>' + esc(icdAxis.label_ru || "МКБ") + '</b>: ' +
        esc(icdAxis.detail_ru || icd.detail_ru || "") +
        '<div class="card-sub">' + esc(icdAxis.note_ru || icd.note_ru || "") + '</div></li>' +
        '<li><b>' + esc(support.label_ru || "Клиническая опора") + '</b>: ' +
        esc(support.band_ru || "") + ' - ' + esc(support.detail_ru || "") + '</li>' +
        '</ul>' +
        (weak ? '<h4>Слабые места методики</h4><ul>' + weak + '</ul>' : "") +
        (gaps ? '<h4>Клиника</h4><ul>' + gaps + '</ul>' : '<p class="card-sub">Клинических разрывов machine не нашёл.</p>') +
        '<h4>МКБ</h4><p>' + esc(icd.detail_ru || "") + '</p>' +
        '<h4>Протокол</h4><p>' + esc(proto.detail_ru || "") +
        (proto.secondary_ru ? '<span class="card-sub"> ' + esc(proto.secondary_ru) + '</span>' : "") +
        '</p>' +
        '<h4>Что сказать врачу</h4>' +
        (feedback ? '<ul id="review-brief-feedback">' + feedback + '</ul>' :
          '<p class="empty">Автопунктов нет.</p>') +
        '<p><button type="button" class="button secondary compact" id="review-brief-prefill">' +
        'Подставить в решение методиста</button></p>' +
        '<p class="card-sub">' + esc((brief.confidence || {}).machine_ru || "") + ' ' +
        esc((brief.confidence || {}).ai_ru || "") + '</p>' +
        ai +
        '</div>';
    }
    function prefillDecisionFromBrief(brief) {
      var area = $("drawer-summary");
      if (!area || !brief) return;
      var text = brief.decision_summary_ru ||
        ((brief.doctor_feedback || []).map(function (line) { return "• " + line; }).join("\n"));
      if (!text) return;
      if (!String(area.value || "").trim()) area.value = text;
    }
    function renderZonesCriteriaDetails(zones) {
      var zoneOrder = [
        ["documentation", "Оформление"],
        ["diagnosis", "Диагноз"],
        ["plan", "План по протоколу"]
      ];
      var all = zones.criteria || [];
      if (!all.length) return "";
      var weak = all.filter(function (c) {
        return c.score === 0 || c.score === 0.5 || (c.score == null && c.na_reason);
      });
      var brief = weak.length
        ? '<ul class="zones-brief-list">' + weak.map(function (item) {
          var mark = item.score === 0 ? "плохо" : (item.score === 0.5 ? "слабо" : "н/д");
          return '<li><b>' + esc(item.title || item.id) + '</b> (' + esc(mark) + ') - ' +
            esc(item.reason || "нужна проверка") + '</li>';
        }).join("") + '</ul>'
        : '<p class="card-sub">По критериям методики замечаний нет (все оценённые пункты = 1).</p>';
      var sections = zoneOrder.map(function (pair) {
        var rows = all.filter(function (c) { return String(c.zone || "") === pair[0]; });
        if (!rows.length) return "";
        var body = rows.map(function (item) {
          var tone = item.score === 1 || item.score === 1.0 ? "good"
            : (item.score === 0.5 ? "review" : (item.score === 0 ? "critical" : "muted"));
          return '<tr class="zones-crit-row zones-crit-row--' + tone + '"><td>' +
            esc(item.title || item.id) + '</td><td><span class="status ' + tone + '">' +
            esc(item.score_label == null ? "н/д" : String(item.score_label)) +
            '</span></td><td>' + esc(item.reason || "") + '</td></tr>';
        }).join("");
        return '<h4 class="zones-crit-group">' + esc(pair[1]) + '</h4>' +
          '<div class="table-wrap"><table class="zones-criteria-table"><thead><tr>' +
          '<th>Параметр</th><th>Оценка</th><th>Пояснение</th></tr></thead><tbody>' +
          body + '</tbody></table></div>';
      }).join("");
      return '<div class="detail-block zones-criteria-block"><h3>Разбор по критериям</h3>' +
        '<p class="card-sub">Методика «Как оценивать»: 1 / 0.5 / 0 / н/д. Сначала слабые места, ниже полная таблица.</p>' +
        brief +
        '<details open class="zones-criteria-details"><summary>Полная таблица критериев</summary>' +
        sections + '</details></div>';
    }
    function renderCase(data) {
      var record = data.record || data.case || data;
      var item = rowRecord(record);
      var axes = data.axes || {};
      var findings = data.findings || record.findings || [];
      var crm = data.crm || record.crm || {};
      var events = data.events || [];
      var packs = data.review_packs || [];
      var coverageInfo = deriveCoverage(data, record, axes);
      var confidenceInfo = deriveConfidence(data, record, axes);
      var sourceDocument = data.document || {};
      var llmJudge = data.llm_action_judge || {};
      var shadowDxPlan = data.shadow_dx_plan || {};
      var zones = data.zones || {};
      var useZonesUi = !!(zones && zones.ok && !zones.skipped);
      var crmStatus = crm.status || "new";
      state.caseDetail = data;
      state.supersedesPackId = "";
      var statusOptions = [
        ["new","Новый"],["assigned","Назначен"],["in_review","На разборе"],
        ["confirmed_issue","Подтверждено"],["false_positive","Отклонено"],
        ["needs_more_data","Нужны данные"],["sent_to_doctor","Передано врачу"],
        ["resolved","Решено"],["closed","Закрыто"]
      ].map(function (option) {
        return '<option value="' + option[0] + '"' + (option[0] === crmStatus ? " selected" : "") + ">" + option[1] + "</option>";
      }).join("");
      $("drawer-title").textContent = "Разбор случая";
      $("drawer-subtitle").textContent = [
        "визит " + (item.visitId || item.id || "-"),
        "пациент " + (item.patientId || "-"),
        item.date, item.doctor, item.specialty, item.diagnosis || ""
      ].filter(Boolean).join(" · ");
      var pdfPath = "/api/methodist/mo/cases/" + encodeURIComponent(item.id) + "/pdf";
      var pdfName = "mo-" + encodeURIComponent(item.id) + ".pdf";
      var decisionHtml =
        '<details class="methodist-decision-panel methodist-decision-panel--dock">' +
        '<summary class="decision-dock-summary">Решение методиста</summary>' +
        '<div class="decision-dock-body">' +
        '<div class="verdict-row">' +
        '<label class="filter"><span>Оформление</span>' + verdictSelect("drawer-verdict-c", "unreviewed") + '</label>' +
        '<label class="filter"><span>Диагноз</span>' + verdictSelect("drawer-verdict-d", "unreviewed") + '</label>' +
        '<label class="filter"><span>План по протоколу</span>' + verdictSelect("drawer-verdict-r", "unreviewed") + '</label>' +
        '</div>' +
        '<label class="filter decision-summary-field"><span>Комментарий врачу</span><textarea class="control" id="drawer-summary" rows="3" maxlength="12000" placeholder="Коротко: что не так и что исправить"></textarea></label>' +
        '<label class="filter"><span>Статус разбора</span><select class="control" id="drawer-status">' + statusOptions + '</select></label>' +
        '<input type="hidden" id="drawer-assignee" value="' + esc(crm.assignee || "") + '">' +
        '<input type="hidden" id="drawer-due" value="' + esc(crm.due_date || "") + '">' +
        '<input type="hidden" id="drawer-tags" value="' + esc((crm.tags || []).join(", ")) + '">' +
        '<details class="mo-secondary-details decision-more"><summary>Дополнительно</summary>' +
        '<label class="filter"><span><input type="checkbox" id="drawer-training-use" checked> Можно использовать для обучения</span></label>' +
        '</details>' +
        '<div class="decision-actions">' +
        '<button class="button" id="drawer-save" type="button">Сохранить</button>' +
        '<button class="button secondary" type="button" data-open-pdf="' + esc(pdfPath) + '" data-open-name="' + esc(pdfName) + '">МО в PDF</button>' +
        '</div></div></details>';
      var drawerPdf = $("drawer-pdf");
      if (drawerPdf) {
        drawerPdf.hidden = false;
        drawerPdf.setAttribute("data-open-pdf", pdfPath);
        drawerPdf.setAttribute("data-open-name", pdfName);
      }
      var reg55Payload = data.reg55 || {};
      var reg55Pct = reg55Payload.reg55_section_pct;
      if (reg55Pct == null) reg55Pct = reg55Payload.regulatory_compliance_pct;
      if (reg55Pct == null) reg55Pct = item.reg55;
      if (reg55Pct == null) reg55Pct = axes.regulatory;
      var reg55BandLabel = reg55Payload.reg55_band_label_ru || "";
      var serviceHtml =
        '<div class="detail-block">' + renderReg55(reg55Payload, reg55Pct) + "</div>" +
        '<details class="detail-block mo-secondary-details"><summary>Служебное: deep, покрытие, CRM</summary>' +
        '<div class="drawer-grid">' + kpi("Сводный индекс", score(data.deep_overall_pct != null ? data.deep_overall_pct : item.total), "deep") +
        kpi("Балл №55", score(reg55Pct), reg55BandLabel || "разд. V · 0/0.5/1") +
        kpi("Полнота проверки", score(coverageInfo.value), "модель") +
        kpi("Надёжность", score(confidenceInfo.value), "модель") + '</div>' +
        renderReviewPackHistory(packs) +
        '<div class="detail-block"><h3>История CRM</h3>' + (events.length ? events.map(function (event) {
          return notice(new Date(event.created_at).toLocaleString("ru-RU"), statusLabel(event.event_type) + " · " + (event.actor || "методист"), "good");
        }).join("") : '<p class="empty">Событий пока нет.</p>') + '</div></details>';
      if (useZonesUi) {
        $("drawer-body").innerHTML =
          '<div class="case-workspace-grid case-workspace-grid--zones">' +
          '<div class="case-workspace-clinical" id="case-clinical-pane">' +
          renderClinicalDocument(sourceDocument, findings) +
          '</div>' +
          '<div class="case-workspace-decision">' +
          '<div class="case-workspace-decision-scroll" id="case-review-pane">' +
          renderZonesHero(zones) +
          renderShadowDxPlan(shadowDxPlan) +
          renderReviewBrief(data.review_brief, data.case_narrative) +
          renderFindingsCompact(findings, crm, llmJudge) +
          renderZonesCriteriaDetails(zones) +
          renderHistoryCompact((function () {
            var hist = data.patient_history || {};
            if (!hist.continuity && data.history_continuity) hist.continuity = data.history_continuity;
            if (!hist.deep && data.history_deep) hist.deep = data.history_deep;
            return hist;
          })()) +
          '<div id="protocol-suggest-host" class="protocol-suggest-host"><p class="card-sub">Подбираем протоколы…</p></div>' +
          serviceHtml +
          '</div>' +
          decisionHtml +
          '</div></div>';
      } else {
        $("drawer-body").innerHTML =
          '<div class="case-workspace-grid"><div class="case-workspace-clinical" id="case-clinical-pane">' +
          renderClinicalDocument(sourceDocument, findings) + serviceHtml +
          '</div><div class="case-workspace-decision">' +
          '<div class="case-workspace-decision-scroll" id="case-review-pane">' +
          renderPatientHistory(data.patient_history) +
          renderShadowDxPlan(shadowDxPlan) +
          renderLlmActionJudge(llmJudge, sourceDocument, item) +
          '<div id="protocol-suggest-host" class="protocol-suggest-host"><p class="card-sub">Подбираем протоколы…</p></div>' +
          renderFindingsCompact(findings, crm, llmJudge) +
          '</div>' + decisionHtml + '</div></div>';
      }
      bindCaseWorkspaceInteractions();
      updateDrawerNav();
      if (useZonesUi) prefillDecisionFromBrief(data.review_brief || {});
      loadProtocolSuggestIntoCase(item.id);
    }
    function bindCaseWorkspaceInteractions() {
      var saveBtn = $("drawer-save");
      if (saveBtn) saveBtn.addEventListener("click", saveCaseDecision);
      var prefillBtn = $("review-brief-prefill");
      if (prefillBtn) {
        prefillBtn.addEventListener("click", function () {
          var brief = (state.caseDetail && state.caseDetail.review_brief) || {};
          var area = $("drawer-summary");
          if (!area) return;
          var text = brief.decision_summary_ru ||
            ((brief.doctor_feedback || []).map(function (line) { return "• " + line; }).join("\n"));
          if (text) area.value = text;
        });
      }
      var body = $("drawer-body");
      if (!body) return;
      body.querySelectorAll("[data-finding-zone]").forEach(function (button) {
        button.addEventListener("click", function () {
          var zone = button.getAttribute("data-finding-zone") || "all";
          body.querySelectorAll("[data-finding-zone]").forEach(function (b) {
            b.classList.toggle("is-active", b === button);
          });
          body.querySelectorAll("[data-zone-filter]").forEach(function (card) {
            card.classList.toggle("is-active", card.getAttribute("data-zone-filter") === zone);
          });
          body.querySelectorAll("[data-finding-zone-item]").forEach(function (card) {
            card.hidden = zone !== "all" && card.getAttribute("data-finding-zone-item") !== zone;
          });
          var findingsBlock = body.querySelector(".findings-compact-list");
          if (findingsBlock) findingsBlock.scrollIntoView({ block: "nearest", behavior: "smooth" });
        });
      });
      body.querySelectorAll("[data-zone-filter]").forEach(function (card) {
        card.addEventListener("click", function () {
          var zone = card.getAttribute("data-zone-filter");
          var btn = body.querySelector('[data-finding-zone="' + zone + '"]');
          if (btn) btn.click();
        });
      });
      body.querySelectorAll("[data-focus-clinical]").forEach(function (button) {
        button.addEventListener("click", function () {
          var field = button.getAttribute("data-focus-clinical");
          var target = body.querySelector('[data-clinical-field="' + field + '"]');
          if (!target) {
            showToast("Поле в тексте МО не найдено");
            return;
          }
          body.querySelectorAll(".clinical-field").forEach(function (node) {
            node.classList.remove("clinical-field--focus");
          });
          target.classList.add("clinical-field--focus");
          target.scrollIntoView({ block: "center", behavior: "smooth" });
        });
      });
      body.querySelectorAll(".patient-history-block [data-case]").forEach(function (button) {
        button.addEventListener("click", function (event) {
          event.preventDefault();
          var cid = button.getAttribute("data-case");
          if (cid) openCase(cid);
        });
      });
      body.querySelectorAll("[data-load-pack]").forEach(function (button) {
        button.addEventListener("click", function () {
          loadReviewPackIntoForm(button.getAttribute("data-load-pack"));
        });
      });
      body.querySelectorAll("[data-retry-protocol-suggest]").forEach(function (button) {
        button.addEventListener("click", function () {
          if (state.openCaseId) loadProtocolSuggestIntoCase(state.openCaseId);
        });
      });
    }
    async function loadReviewPackIntoForm(packId) {
      if (!packId) return;
      try {
        var response = await request("/review-packs/" + encodeURIComponent(packId));
        if (!response.ok) throw new Error("Не удалось открыть пакет разбора.");
        var payload = await response.json();
        var pack = payload.pack || {};
        var decision = pack.decision || {};
        state.supersedesPackId = pack.pack_id || packId;
        if ($("drawer-status") && decision.status) $("drawer-status").value = decision.status;
        if ($("drawer-assignee")) $("drawer-assignee").value = decision.assignee || "";
        if ($("drawer-due")) $("drawer-due").value = decision.due_date || "";
        if ($("drawer-tags")) $("drawer-tags").value = (decision.tags || []).join(", ");
        if ($("drawer-verdict-c")) $("drawer-verdict-c").value = decision.verdict_completeness || "unreviewed";
        if ($("drawer-verdict-d")) $("drawer-verdict-d").value = decision.verdict_diagnosis || "unreviewed";
        if ($("drawer-verdict-r")) $("drawer-verdict-r").value = decision.verdict_recommendations || "unreviewed";
        if ($("drawer-summary")) $("drawer-summary").value = decision.summary_ru || "";
        if ($("drawer-training-use")) $("drawer-training-use").checked = decision.training_use !== false;
        Object.keys(decision.finding_decisions || {}).forEach(function (code) {
          var select = $("drawer-body").querySelector('[data-finding-code="' + code + '"]');
          if (select) select.value = decision.finding_decisions[code];
        });
        $("announcer").textContent = "Пакет загружен для правки - сохранение создаст новую версию";
        var panel = $("drawer-body").querySelector(".methodist-decision-panel");
        if (panel) panel.scrollIntoView({ block: "nearest", behavior: "smooth" });
      } catch (e) { showError(e.message); }
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
        var why = (rubric && (rubric.reason || rubric.error)) ?
          String(rubric.reason || rubric.error) :
          "Shadow-оценка по методике МЗ пока недоступна для этого случая.";
        return '<div class="detail-block"><h3>Рубрика МЗ («Как оценивать»)</h3>' +
          '<p class="empty">' + esc(why) + '</p>' +
          '<p class="card-sub">Рубрика МЗ («Как оценивать», шкала 0 / 0.5 / 1) - отдельно от баллов №55. ' +
          '№55 проверяет пункты постановления pass/fail; рубрика - полноту записи по инструкции №127.</p></div>';
      }
      var groupLabels = {
        documentation: "Документация", clinical: "Клиника",
        dynamics: "Динамика", regulatory: "Регламент"
      };
      var rows = (rubric.criteria || []).map(function (item) {
        var label = item.score_label == null ? "нет данных" : String(item.score_label);
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
        ', нет данных ' + esc(rubric.na_n != null ? rubric.na_n : " - ") +
        ' · итог ' + esc(score(rubric.rubric_pct)) +
        (rubric.prior_available ? ' · предыдущий визит ' + esc(rubric.prior_visit_date || "") : ' · предыдущий визит: нет') +
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
      return '<div class="detail-block clinical-mo-block"><h3>Текст МО</h3>' +
        '<p class="card-sub">Слева читаете запись. Справа - оценки и решение. Подсветка «↔ замечание» связана с блоком «Что не так».</p>' +
        reason +
        (content || '<div class="empty"><b>Клинический текст недоступен</b><div>Нет опубликованного secure CSV/parquet за дату визита. Повторите publish или откройте визит в МИС.</div></div>') +
        '<p class="card-sub">Источник: ' + esc(sourceLabel) + '</p></div>';
    }
    async function postCaseChanges(caseIds, changes, comment) {
      var response = await request("/cases/bulk-action", "/cases/bulk-action", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ case_ids: caseIds, changes: changes, comment: comment || "" })
      });
      if (!response.ok) throw new Error("Не удалось сохранить изменения.");
      return response.json();
    }
    function collectProtocolRatings() {
      var ratings = [];
      document.querySelectorAll(".protocol-suggest-item[data-protocol-id]").forEach(function (node) {
        var pid = node.getAttribute("data-protocol-id");
        var checked = node.querySelector('input[type="radio"]:checked');
        var titleEl = node.querySelector("b");
        ratings.push({
          protocol_id: pid,
          title: titleEl ? titleEl.textContent.replace(/^\d+\.\s*/, "") : "",
          relevance: checked ? checked.value : "unreviewed"
        });
      });
      return ratings;
    }
    async function loadProtocolSuggestIntoCase(caseId) {
      var host = $("protocol-suggest-host");
      if (!host || !caseId) return;
      host.innerHTML = '<div class="detail-block"><p class="card-sub">Подбираем протоколы…</p></div>';
      try {
        var q = query();
        q.set("month", q.get("month") || minskDateKey(0).slice(0, 7));
        var response = await request(
          "/cases/" + encodeURIComponent(caseId) + "/protocol-suggest?" + q.toString()
        );
        if (!response.ok) throw new Error("suggest_failed");
        var suggest = await response.json();
        host.innerHTML = renderProtocolSuggest(suggest);
        bindProtocolSuggestHost(host);
        if (state.caseDetail) {
          state.caseDetail.protocol_suggest = suggest;
          if (state.caseDetail.review_brief && suggest && suggest.available) {
            var items = suggest.items || [];
            var top = items[0] || {};
            state.caseDetail.review_brief.protocol = {
              matched: true,
              kp_status: "matched",
              title: top.title || null,
              protocol_id: top.protocol_id || null,
              detail_ru: top.title
                ? ("Топ-1: " + top.title + (items.length > 1 ? (" · ещё " + (items.length - 1)) : ""))
                : "Протоколы подобраны",
              items_n: items.length
            };
          }
        }
      } catch (e) {
        host.innerHTML = renderProtocolSuggest({ available: false, reason: "Нет клинического протокола МЗ по этому диагнозу" });
        bindProtocolSuggestHost(host);
      }
    }
    function updateDrawerNav() {
      var ids = state.caseNavIds || [];
      var idx = ids.indexOf(state.openCaseId);
      var prev = $("drawer-prev");
      var next = $("drawer-next");
      if (!prev || !next) return;
      if (idx < 0 || ids.length < 2) {
        prev.hidden = true; next.hidden = true; return;
      }
      prev.hidden = idx <= 0;
      next.hidden = idx >= ids.length - 1;
      prev.disabled = idx <= 0;
      next.disabled = idx >= ids.length - 1;
    }
    async function saveCaseDecision() {
      try {
        var findingDecisions = {};
        document.querySelectorAll("[data-finding-code]").forEach(function (select) {
          findingDecisions[select.getAttribute("data-finding-code")] = select.value;
        });
        var summaryText = ($("drawer-summary") && $("drawer-summary").value) || "";
        if (summaryText.trim().length && summaryText.trim().length < 80) {
          if (!window.confirm("Развёрнутый разбор короткий (меньше 80 символов). Сохранить всё равно?")) return;
        }
        var trainingUse = !($("drawer-training-use") && !$("drawer-training-use").checked);
        var findingValues = Object.keys(findingDecisions).map(function (key) { return findingDecisions[key]; });
        var allFindingsUnreviewed = findingValues.length > 0 && findingValues.every(function (value) {
          return !value || value === "unreviewed";
        });
        if (trainingUse && allFindingsUnreviewed) {
          if (!window.confirm(
            "Для обучения отмечены все замечания как «не просмотрено». Сохранить с флагом обучения всё равно?"
          )) return;
        }
        var decision = {
          status: $("drawer-status").value,
          assignee: $("drawer-assignee").value.trim(),
          due_date: $("drawer-due").value,
          tags: $("drawer-tags").value.split(",").map(function (tag) { return tag.trim(); }).filter(Boolean),
          finding_decisions: findingDecisions,
          verdict_completeness: ($("drawer-verdict-c") && $("drawer-verdict-c").value) || "unreviewed",
          verdict_diagnosis: ($("drawer-verdict-d") && $("drawer-verdict-d").value) || "unreviewed",
          verdict_recommendations: ($("drawer-verdict-r") && $("drawer-verdict-r").value) || "unreviewed",
          summary_ru: summaryText,
          training_use: trainingUse,
          protocol_ratings: collectProtocolRatings(),
          protocol_suggest: state.protocolSuggest || null
        };
        var body = { decision: decision };
        if (state.supersedesPackId) body.supersedes_pack_id = state.supersedesPackId;
        var month = (query().get("month") || minskDateKey(0).slice(0, 7));
        if (month) body.month = month;
        var response = await request(
          "/cases/" + encodeURIComponent(state.openCaseId) + "/review-pack",
          null,
          { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
        );
        if (!response.ok) throw new Error("Не удалось сохранить пакет разбора.");
        $("announcer").textContent = "Пакет разбора сохранён";
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
      if ($("drawer-pdf")) $("drawer-pdf").hidden = true;
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
      attachTableChrome($("yesterday-kind-rows").closest("table"), { id: "chrome-yesterday-kind-rows" });
    }
    function renderYesterdayIndices(data) {
      if (!hostActive("yesterday-index-cards")) return;
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
      if (!hostActive("yesterday-findings-chart")) return;
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
          '<span class="status ' + esc(severityTone(item)) + '">' +
          esc(severityLabel(item)) + '</span> <b>' + esc(item.label || item.finding_code) + '</b>' +
          '<span class="finding-meta">' + esc(item.cases) + ' случаев · открыть список МО</span></button>' +
          (samples ? '<div class="finding-cases">' + samples + '</div>' : '') +
          '</div>';
      }).join("");
    }
    function renderYesterdayDoctors(data) {
      if (!hostActive("yesterday-doctor-chart")) return;
      var section = data.doctor_outliers || {}, items = section.items || [];
      if (!items.length) {
        $("yesterday-doctor-chart").innerHTML = unavailableBlock(section);
        $("yesterday-doctor-note").innerHTML = "";
        return;
      }
      $("yesterday-doctor-note").innerHTML = section.note
        ? notice("Ожидаемое", section.note, "review")
        : "";
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
    function llmJudgeMini(item) {
      var judge = item.llm_action_judge || {};
      if (!judge.available || !judge.kpis) return "";
      var k = judge.kpis;
      function chip(label, payload) {
        var pct = payload && payload.score_pct;
        var tone = verdictTone(payload && payload.verdict);
        return '<span class="status ' + tone + ' llm-mini-chip" title="' + esc(label) + '">' +
          esc(label[0]) + " " + esc(pct == null ? "-" : Math.round(Number(pct)) + "%") + '</span>';
      }
      return '<div class="llm-mini-kpis">' +
        chip("Полнота", k.completeness) + chip("Диагноз", k.diagnosis) + chip("Рекомендации", k.recommendations) +
        '</div>';
    }
    function renderYesterdayActions(data) {
      var section = data.action_cases || {}, items = section.items || [];
      $("yesterday-action-rows").innerHTML = items.length ? items.map(function (item) {
        var pdfUrl = item.pdf_url || ("/api/methodist/mo/cases/" + encodeURIComponent(item.case_id) + "/pdf");
        var visitId = item.visit_id || item.case_id || "-";
        var layer = item.layer_ru || layerLabelRu(item.attention_primary) || "-";
        var reason = item.attention_reason_ru || item.reason || item.finding_title || item.finding_code || "";
        var deep = item.deep_run_track_ru || item.history_mode_ru || "";
        if (deep) reason = (reason ? reason + " · " : "") + deep;
        return '<tr data-case="' + esc(item.case_id) + '"><td><span class="status ' +
          esc(severityTone(item)) + '">' + esc(severityLabel(item)) +
          '</span></td><td>' + esc(layer) +
          '</td><td class="id-cell">' + esc(visitId) +
          '</td><td class="id-cell">' + esc(item.patient_id || "-") +
          '</td><td>' + esc(item.visit_date || data.date || "-") +
          '</td><td><b>' + esc(item.doctor_fio || item.doctor) + "</b><br><small>" + esc(item.specialty) +
          "</small>" +
          "</td><td>" + esc(item.filial || item.branch) + "</td><td>" + esc(item.diagnosis) +
          "</td><td>" + esc(reason) +
          '</td><td class="row-actions"><button class="button secondary compact" type="button" data-open-pdf="' + esc(pdfUrl) + '" data-open-name="mo-' + esc(item.case_id) + '.pdf">МО в PDF</button></td></tr>';
      }).join("") : '<tr><td colspan="10">' + unavailableBlock(section, "Случаев для разбора нет.") + "</td></tr>";
      bindCaseRows($("yesterday-action-rows"));
      attachTableChrome($("yesterday-action-rows").closest("table"), { id: "chrome-yesterday-action-rows" });
    }
    function renderYesterdayFlow(data, dimension) {
      if (!hostActive("yesterday-flow-chart")) return;
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
      if (!hostActive("yesterday-source-quality")) return;
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
    function renderYesterday(data, dash, workingDay) {
      renderYesterdayScoreKpis(data, dash || null);
      renderAttentionStrip("yesterday-attention", data.attention || null);
      renderYesterdayScoreDashboard(dash || state.data.scoreDashboard || null, workingDay || data.date || "");
      renderYesterdayActions(data);
      renderYesterdayCompleteness(data);
      renderYesterdayIndices(data);
      renderYesterdayFindings(data);
      renderYesterdayDoctors(data);
      var flowDim = $("yesterday-flow-dimension");
      renderYesterdayFlow(data, (flowDim && flowDim.value) || "specialty");
      renderYesterdaySourceQuality(data);
    }
    async function resolveTodayWorkingDay() {
      if (state.period === "custom" && state.dateFrom) {
        return { day: state.dateFrom, fallback: false, preferred: state.dateFrom };
      }
      var preferred = minskDateKey(-1);
      var through = "";
      try {
        var fr = await request("/freshness", "/freshness");
        if (fr.ok) {
          var fj = await fr.json();
          through = String(fj.data_through || "").slice(0, 10);
        }
      } catch (e) {}
      if (through && through < preferred) {
        return { day: through, fallback: true, preferred: preferred };
      }
      return { day: preferred, fallback: false, preferred: preferred };
    }
    async function loadYesterday() {
      var resolved = await resolveTodayWorkingDay();
      var day = resolved.day;
      var label = "Итоги за " + new Date(day + "T12:00:00").toLocaleDateString("ru-RU", { dateStyle:"long" }) + ".";
      if (resolved.fallback) {
        label += " Показан последний день с данными (за " +
          new Date(resolved.preferred + "T12:00:00").toLocaleDateString("ru-RU", { dateStyle:"medium" }) +
          " выгрузки ещё нет).";
      }
      $("yesterday-date").textContent = label;
      var dashPromise = request("/score-dashboard?" + query().toString(), "/score-dashboard");
      var response = await request("/daily-report?date=" + encodeURIComponent(day), "__root__");
      if (await handleHttpAuth(response)) return;
      if (!response.ok) throw new Error("Отчёт за " + day + " пока недоступен.");
      var data = await response.json();
      var hasSignal = !!(data.attention && (data.attention.n_evaluated || data.attention.n_evaluated === 0));
      if ((!hasSignal || !(data.attention && data.attention.n_evaluated)) && !resolved.fallback) {
        // empty calendar day with ok:true - try freshness fallback once
        try {
          var fr2 = await request("/freshness", "/freshness");
          if (fr2.ok) {
            var through2 = String((await fr2.json()).data_through || "").slice(0, 10);
            if (through2 && through2 !== day) {
              resolved = { day: through2, fallback: true, preferred: day };
              day = through2;
              $("yesterday-date").textContent = "Итоги за " +
                new Date(day + "T12:00:00").toLocaleDateString("ru-RU", { dateStyle:"long" }) +
                ". Показан последний день с данными (свежих выгрузок пока нет).";
              response = await request("/daily-report?date=" + encodeURIComponent(day), "__root__");
              if (!response.ok) throw new Error("Отчёт за " + day + " пока недоступен.");
              data = await response.json();
            }
          }
        } catch (e) {}
      }
      var dash = { ok: false, available: false, reason: "Сводка оценок недоступна." };
      try {
        var dashResp = await dashPromise;
        if (dashResp && dashResp.ok) dash = await dashResp.json();
        else if (dashResp && dashResp.status === 404) {
          dash.reason = "API score-dashboard ещё не на сервере.";
        }
      } catch (e) {
        dash.reason = "Не удалось загрузить кольца и динамику.";
      }
      state.data.daily = data;
      state.data.scoreDashboard = dash;
      var fresh = $("yesterday-freshness");
      if (fresh) fresh.textContent = "Данные по " + (data.data_through || data.date || day);
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
      renderYesterday(data, dash, day);
    }
    function renderEntityPages(summary) {
      if ($("diagnosis-findings")) {
        $("diagnosis-findings").innerHTML = (summary.findings || []).slice(0,8).map(function (x) {
          return notice(severityLabel(x) || "Проверить", x.title || x.label || "Требуется ручная проверка", severityTone(x));
        }).join("") || '<div class="empty">Замечаний по выбранному срезу нет.</div>';
      }
      if (!$("quality-kpis")) return;
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
    function pctOrDash(value) {
      return value == null || value === "" ? "-" : (Number(value).toFixed(1).replace(/\.0$/, "") + "%");
    }
    function openDoctorCases(item, zoneKey) {
      zoneKey = zoneKey || state.doctorZoneMetric || "zone1";
      applyDrill({
        label: "Врач " + (item.label || item.key),
        selected: { doctors: [item.label || item.key] },
        zoneFilter: zoneKey,
        zoneBandFilter: "bad",
        attentionOnly: false,
        page: "documents"
      });
    }
    function renderDoctorZoneChart(items) {
      var metric = state.doctorZoneMetric || "zone1";
      var pctKey = metric + "_bad_pct";
      var ranked = items.filter(function (x) {
        return !x.suppressed && x[pctKey] != null && Number(x.n || 0) >= 5;
      }).slice().sort(function (a, b) {
        return Number(b[pctKey] || 0) - Number(a[pctKey] || 0);
      }).slice(0, 20).reverse();
      var host = $("doctor-zone-chart");
      if (!host) return;
      if (!ranked.length) {
        host.innerHTML = '<p class="empty">Нет данных по зонам за период (нужен recompute после деплоя) или выборка меньше порога.</p>';
        return;
      }
      var chart = MO.moChart(host, {
        tooltip: {
          trigger: "axis",
          formatter: function (params) {
            var p = params && params[0];
            if (!p) return "";
            var row = ranked[p.dataIndex];
            return esc(row.label) + "<br>" + esc(ZONE_LABELS[metric] || metric) +
              " плохо: " + pctOrDash(row[pctKey]) + "<br>Случаев: " + esc(row.n);
          }
        },
        grid: { left: 160, right: 28, top: 18, bottom: 36 },
        xAxis: { type: "value", name: "% плохо", max: 100 },
        yAxis: { type: "category", data: ranked.map(function (x) { return x.label; }) },
        series: [{
          type: "bar",
          barMaxWidth: 16,
          itemStyle: { borderRadius: [0, 6, 6, 0], color: cssToken("--bad", "#9a5b66") },
          data: ranked.map(function (x) { return Number(x[pctKey] || 0); })
        }]
      }, {
        label: "Доля плохого: " + (ZONE_LABELS[metric] || metric),
        description: "Клик по полосе открывает случаи врача с фильтром «плохо» по выбранному разделу."
      });
      if (chart) {
        chart.on("click", function (params) {
          var row = ranked[params.dataIndex];
          if (row) openDoctorCases(row, metric);
        });
      }
      var toggle = $("doctor-zone-metric");
      if (toggle) {
        toggle.querySelectorAll("[data-doctor-zone]").forEach(function (btn) {
          btn.setAttribute("aria-pressed", btn.getAttribute("data-doctor-zone") === metric ? "true" : "false");
        });
      }
    }
    async function loadDoctorsDimension() {
      var data = await dimensionData("doctors"), items = data.items || [];
      state.data.doctorItems = items;
      $("doctor-rows").innerHTML = items.length ? items.map(function (x) {
        return '<tr data-doctor-key="' + esc(x.key) + '">' +
          "<td><b>" + esc(x.label) + "</b></td><td>" + esc(x.specialty) +
          "</td><td>" + esc(x.n == null ? x.n_bucket : x.n) +
          "</td><td>" + esc(pctOrDash(x.zone1_bad_pct)) +
          "</td><td>" + esc(pctOrDash(x.zone2a_bad_pct)) +
          "</td><td>" + esc(pctOrDash(x.zone2b_bad_pct)) +
          "</td><td>" + esc(x.attention_n == null ? "-" : x.attention_n) +
          '</td><td><button class="button secondary compact" type="button" data-open-doctor-cases="' +
          esc(x.label) + '" data-doctor-key="' + esc(x.key) + '">Открыть случаи</button></td></tr>';
      }).join("") : '<tr><td colspan="8" class="empty">Нет данных по врачам.</td></tr>';
      $("doctor-rows").querySelectorAll("[data-open-doctor-cases]").forEach(function (button) {
        button.addEventListener("click", function () {
          var label = button.getAttribute("data-open-doctor-cases") || "";
          var key = button.getAttribute("data-doctor-key") || label;
          openDoctorCases({ label: label, key: key }, state.doctorZoneMetric);
        });
      });
      attachTableChrome($("doctor-rows").closest("table"), { id: "chrome-doctor-rows" });
      renderDoctorZoneChart(items);
      var plotted = items.filter(function (x) { return x.enough_data && !x.suppressed && x.delta != null; });
      var scatterHost = $("doctor-scatter-chart");
      if (!scatterHost) return;
      var chart = MO.moChart(scatterHost, {
        tooltip:{ formatter:function (p) { var x=plotted[p.dataIndex], ci=x.delta_ci95 || {};
          return esc(x.label)+"<br>Объём: "+x.n+"<br>Дельта: "+signed(x.delta)+
            "<br>95% ДИ: "+signed(ci.low)+" - "+signed(ci.high)+"<br>Критично: "+(x.p0_cases || 0); } },
        toolbox:{ feature:{ brush:{ type:["rect","clear"] }, dataZoom:{}, saveAsImage:{} } },
        brush:{ toolbox:["rect","clear"], xAxisIndex:"all", yAxisIndex:"all" },
        grid:{ left:58,right:30,top:55,bottom:55 },
        xAxis:{ type:"value", name:"Число записей" },
        yAxis:{ type:"value", name:"Дельта к ожидаемой, п.п.", axisLine:{ onZero:true } },
        series:[{ type:"scatter", data:plotted.map(function (x) {
          return { value:[x.n,x.delta,Math.max(8,Math.min(42,8+(x.p0_cases || 0)*4))], doctor:x };
        }), symbolSize:function (value) { return value[2]; } }]
      }, { label:"Врачи: объём и дельта к ожидаемой оценке",
        description:"Дополнительный разрез. Основной экран - таблица зон и полосы «плохо»." });
      if (chart) {
        chart.on("click",function (params) {
          if (!plotted[params.dataIndex]) return;
          openDoctorCases(plotted[params.dataIndex], state.doctorZoneMetric);
        });
        chart.on("brushSelected",function (params) {
          var selected=[], batches=(params.batch && params.batch[0] && params.batch[0].selected) || [];
          batches.forEach(function (batch) { (batch.dataIndex || []).forEach(function (index) {
            if (plotted[index] && selected.indexOf(plotted[index]) < 0) selected.push(plotted[index]);
          }); });
          $("doctor-selection-flow").innerHTML=selected.length ?
            "<p><b>Выбрано врачей: "+selected.length+"</b></p><p>"+selected.map(function (x) { return esc(x.label); }).join(", ")+
            '</p><button class="button" id="open-selected-doctors">Открыть их случаи</button>' :
            "Выделите точки рамкой.";
          var action=$("open-selected-doctors");
          if (action) action.addEventListener("click",function () {
            applyDrill({ label: "Группа врачей", selected: { doctors: selected.map(function (x) { return x.label; }) },
              zoneFilter: state.doctorZoneMetric, zoneBandFilter: "bad", page: "documents" });
          });
        });
      }
    }
    function applyZonePreset(key) {
      var preset = ZONE_PRESETS[key];
      if (!preset) return;
      state.zoneFilter = preset.zoneFilter || "";
      state.zoneBandFilter = preset.zoneBandFilter || "";
      state.attentionOnly = !!preset.attentionOnly;
      state.kpStatus = preset.kpStatus || "";
      state.historyTier = preset.historyTier || "";
      state.pageNo = 1;
      renderChips();
      syncUrl(true);
      switchPage(preset.page || "documents", false);
      showToast(preset.name);
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
      var levelLabels = { P0: "Критично", P1: "Важно", P2: "Умеренно", P3: "Оформление" };
      $("safety-kpis").innerHTML=levels.map(function (level) {
        return kpi(levelLabels[level] || level,items.reduce(function (sum,row) { return sum+(row[level] || 0); },0),"случаев с замечанием");
      }).join("");
      var incidents=data.incidents || [];
      var legend = levels.map(function (level) { return levelLabels[level] || level; });
      MO.moChart($("safety-severity-chart"),{
        tooltip:{ trigger:"axis" },legend:{ data:legend },grid:{ left:50,right:25,top:50,bottom:55 },
        xAxis:{ type:"category",data:items.map(function (x) { return x.date; }) },yAxis:{ type:"value",name:"Случаи" },
        series:levels.map(function (level) { return { name:levelLabels[level] || level,type:"bar",stack:"severity",
          data:items.map(function (x) { return x[level] || 0; }),
          markPoint:level==="P0" ? { data:incidents.map(function (x) {
            return { name:x.finding_code,coord:[x.date,0],value:"!" };
          }) } : undefined }; })
      },{ label:"Замечания по приоритету по дням",description:"Столбцы сложены по приоритету; маркеры - случаи «Критично»." });
      $("safety-list").innerHTML=incidents.slice(0,30).map(function (x) {
        return notice("Критично · "+x.finding_code,x.date+" · источник: "+(x.source_ref || "не указан"),"critical");
      }).join("") || '<div class="empty">Инцидентов «Критично» в выбранном периоде нет.</div>';
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
      var host=$("access-log-content"); if (!host) return;
      var response=await request("/access-log","/access-log");
      if (response.status===403) { host.innerHTML='<div class="empty">Журнал доступен только администратору.</div>'; return; }
      if (!response.ok) throw new Error("Не удалось загрузить журнал доступа.");
      var data=await response.json();
      host.innerHTML=(data.items || []).map(function (item) {
        return notice(item.action,item.created_at+" · "+item.actor+" · роль "+item.role+
          (item.doctor_key ? " · врач "+item.doctor_key : ""),"good");
      }).join("") || '<div class="empty">Событий доступа пока нет.</div>';
    }
    async function ensureSummary() {
      if (!state.data.summary) await loadOverview();
      renderEntityPages(state.data.summary);
    }
    async function loadDataQuality() {
      if (!$("quality-kpis")) return;
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
        document.querySelectorAll(".nav-section-label").forEach(function (label) {
          var anyVisible = false;
          var next = label.nextElementSibling;
          while (next && !next.classList.contains("nav-section-label")) {
            if (!next.hidden) anyVisible = true;
            next = next.nextElementSibling;
          }
          label.hidden = !anyVisible;
        });
        if (isExpertMode()) {
          document.querySelectorAll('[data-go="queue"]').forEach(function (el) { el.hidden = true; });
          document.querySelectorAll('[data-action="export-aggregates"], [data-action="export"], #print-report, #open-briefing-html').forEach(function (el) {
            el.hidden = true;
          });
          var minDate = capabilities.reports_min_date;
          if (minDate && $("title-reports")) {
            var sub = $("page-reports") && $("page-reports").querySelector(".page-head p");
            if (sub) sub.textContent = "Ежедневные отчёты с " + minDate + ". Клик открывает день на экране «Вчера».";
          }
          if ($("expert-user-label")) {
            $("expert-user-label").textContent = state.expertDisplayName || capabilities.role || "Эксперт";
          }
        }
      } catch (error) {}
    }
    async function loadReports() {
      var jobs = [
        request("/reports", "/dynamics"),
        request("/freshness?" + query().toString(), "/dynamics")
      ];
      if (!isExpertMode()) jobs.push(request("/month-report?" + query().toString(), "__root__"));
      var responses = await Promise.all(jobs);
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
          kpi("Отчётов в списке", items.length, isExpertMode() ? ("с " + (data.min_date || "2026-08-01")) : "ежедневные готовые срезы") +
          (isExpertMode()
            ? kpi("Роль", "Эксперт", "разбор и обучение")
            : kpi("Оценено за месяц", ((month.kpi || {}).evaluated != null ? month.kpi.evaluated : "н/д"),
              score((month.kpi || {}).avg_score)));
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
          applyDrill({ label: "Таблица за " + day, period: "custom", dateFrom: day, dateTo: day, page: "documents" });
        });
      });
    }
    function currentReportDay() {
      if (state.period === "custom" && state.dateFrom) return state.dateFrom;
      return minskDateKey(-1);
    }
    function accessLabelRu(value) {
      if (value === "full") return "полная МО Аналитика";
      if (value === "reports") return "только отчёты";
      return value || "не указан";
    }
    function roleLabelRu(value) {
      if (value === "admin") return "админ";
      if (value === "viewer") return "просмотр";
      if (value === "methodist") return "методист";
      return value || "методист";
    }
    var scoringStrictnessPollTimer = null;
    function scoringStrictnessCanEdit() {
      var actions = (state.data.capabilities && state.data.capabilities.actions) || {};
      return actions.manage_scoring_config !== false;
    }
    function stopScoringStrictnessPoll() {
      if (scoringStrictnessPollTimer) {
        window.clearInterval(scoringStrictnessPollTimer);
        scoringStrictnessPollTimer = null;
      }
    }
    function startScoringStrictnessPoll() {
      stopScoringStrictnessPoll();
      var ticks = 0;
      scoringStrictnessPollTimer = window.setInterval(async function () {
        ticks += 1;
        try {
          var response = await request("/scoring-config", "/scoring-config");
          if (!response.ok) return;
          var data = await response.json();
          var job = data.recompute_job || {};
          renderScoringStrictness(data);
          if (!job.status || job.status === "done" || job.status === "error" || ticks >= 40) {
            stopScoringStrictnessPoll();
            var el = $("score-strictness-status");
            if (el && job.status === "done") el.textContent = "Пересчёт витрины завершён.";
            if (el && job.status === "error") {
              el.textContent = "Пересчёт завершился с ошибкой.";
              el.style.color = "var(--danger, #b42318)";
            }
          }
        } catch (error) {}
      }, 2000);
    }
    function renderScoringStrictness(payload) {
      var host = $("scoring-strictness");
      if (!host) return;
      var prevFrom = $("score-date-from") ? $("score-date-from").value : "";
      var prevTo = $("score-date-to") ? $("score-date-to").value : "";
      var profile = (payload && payload.profile) || {};
      var effective = (payload && payload.effective) || {};
      var days = (payload && payload.available_days) || {};
      var job = (payload && payload.recompute_job) || null;
      var notes = (payload && payload.notes_ru) || [];
      var zb = effective.zone_bands || profile.zone_bands || {};
      var st = effective.status_thresholds || profile.status_thresholds || {};
      var caps = effective.risk_caps || profile.risk_caps || {};
      var preset = profile.preset || "standard";
      var canEdit = scoringStrictnessCanEdit();
      var disabled = canEdit ? "" : " disabled";
      var jobHtml = "";
      if (job && job.status) {
        jobHtml =
          '<p class="card-sub">Пересчёт витрины/зон: <b>' + esc(job.status) + "</b>" +
          (job.date_from ? " · " + esc(job.date_from) + " - " + esc(job.date_to || "") : "") +
          (job.progress ? " · " + esc(job.progress.done) + "/" + esc(job.progress.total) : "") +
          "</p>";
      }
      var notesHtml = notes.length
        ? '<ul class="card-sub" style="margin:8px 0 0 1.1rem;padding:0">' +
          notes.map(function (n) { return "<li>" + esc(n) + "</li>"; }).join("") +
          "</ul>"
        : "";
      host.innerHTML =
        '<div class="settings-stack scoring-strictness-form">' +
        '<p class="card-sub">Меняет полосы зон и пороги внимания. Пересчёт витрины обновляет зоны без повторного LLM; deep-пересчёт переписывает findings.</p>' +
        '<label class="filter"><span>Пресет жёсткости</span>' +
        '<select class="control" id="score-preset"' + disabled + ">" +
        '<option value="soft">Мягкая</option>' +
        '<option value="standard">Стандарт</option>' +
        '<option value="strict">Жёсткая</option>' +
        '<option value="custom">Своя</option>' +
        "</select></label>" +
        '<div class="grid">' +
        '<label class="filter span-3"><span>Зона «плохо» ниже, %</span><input class="control" id="score-bad-below" type="number" min="0" max="100" step="1"' + disabled + "></label>" +
        '<label class="filter span-3"><span>Зона «в норме» от, %</span><input class="control" id="score-ok-at" type="number" min="0" max="100" step="1"' + disabled + "></label>" +
        '<label class="filter span-3"><span>Статус «хорошо» от</span><input class="control" id="score-good" type="number" min="0" max="100" step="1"' + disabled + "></label>" +
        '<label class="filter span-3"><span>Статус «приемлемо» от</span><input class="control" id="score-acc" type="number" min="0" max="100" step="1"' + disabled + "></label>" +
        '<label class="filter span-3"><span>Потолок при «Критично»</span><input class="control" id="score-p0" type="number" min="0" max="100" step="1"' + disabled + "></label>" +
        '<label class="filter span-3"><span>Потолок при «Важно»</span><input class="control" id="score-p1" type="number" min="0" max="100" step="1"' + disabled + "></label>" +
        '<label class="filter span-3"><span>В очередь при балле ниже</span><input class="control" id="score-attention" type="number" min="0" max="100" step="1"' + disabled + "></label>" +
        "</div>" +
        '<p class="card-sub">Потолки ограничивают итоговый балл, если замечание уже выставлено. Не меняют правила, какое замечание считать Важно или Критично (это не таксономия).</p>' +
        '<p class="card-sub">Доступные дни витрины: ' +
        (days.n ? (esc(days.first) + " - " + esc(days.last) + " (" + esc(days.n) + ")") : "пока нет") +
        ". Версия профиля: " + esc(profile.profile_version || 1) +
        (profile.apply_on_next_load ? " · ждёт следующую загрузку" : "") +
        "</p>" +
        jobHtml +
        notesHtml +
        (canEdit
          ? '<div class="section-actions" style="margin-top:8px">' +
            '<button class="button" type="button" id="score-save">Сохранить</button>' +
            '<button class="button secondary" type="button" id="score-recompute-range">Пересчитать период</button>' +
            '<button class="button secondary" type="button" id="score-recompute-all">Пересчитать всё</button>' +
            '<button class="button secondary" type="button" id="score-deep-rescore-range">Deep-пересчёт периода</button>' +
            '<button class="button secondary" type="button" id="score-next-load-days">На след. загрузку (новые дни)</button>' +
            '<button class="button secondary" type="button" id="score-next-load-range">На след. загрузку (период)</button>' +
            '<button class="button secondary" type="button" id="score-next-load-all">На след. загрузку (вся история)</button>' +
            "</div>" +
            '<div class="grid" style="margin-top:8px">' +
            '<label class="filter span-4"><span>Период с</span><input class="control" id="score-date-from" type="date"></label>' +
            '<label class="filter span-4"><span>по</span><input class="control" id="score-date-to" type="date"></label>' +
            "</div>" +
            '<p class="card-sub" id="score-strictness-status"></p>'
          : '<p class="card-sub">Изменение жёсткости доступно методисту и админу.</p>') +
        "</div>";
      if ($("score-preset")) $("score-preset").value = preset;
      if ($("score-bad-below")) $("score-bad-below").value = zb.bad_below != null ? zb.bad_below : 50;
      if ($("score-ok-at")) $("score-ok-at").value = zb.ok_at_or_above != null ? zb.ok_at_or_above : 85;
      if ($("score-good")) $("score-good").value = st.good != null ? st.good : 78;
      if ($("score-acc")) $("score-acc").value = st.acceptable != null ? st.acceptable : 58;
      if ($("score-p0")) $("score-p0").value = caps.P0 != null ? caps.P0 : 40;
      if ($("score-p1")) $("score-p1").value = caps.P1 != null ? caps.P1 : 60;
      if ($("score-attention")) $("score-attention").value = profile.attention_score_below != null ? profile.attention_score_below : 70;
      if ($("score-date-from")) $("score-date-from").value = prevFrom || days.first || "";
      if ($("score-date-to")) $("score-date-to").value = prevTo || days.last || "";
      if (!canEdit) return;
      var presets = (profile.presets) || {};
      if ($("score-preset")) {
        $("score-preset").addEventListener("change", function () {
          var key = $("score-preset").value;
          var pack = presets[key];
          if (!pack) return;
          $("score-bad-below").value = pack.zone_bands.bad_below;
          $("score-ok-at").value = pack.zone_bands.ok_at_or_above;
          $("score-good").value = pack.status_thresholds.good;
          $("score-acc").value = pack.status_thresholds.acceptable;
          $("score-p0").value = pack.risk_caps.P0;
          $("score-p1").value = pack.risk_caps.P1;
          $("score-attention").value = pack.attention_score_below;
        });
      }
      function collectPatch() {
        return {
          preset: $("score-preset").value,
          zone_bands: {
            bad_below: Number($("score-bad-below").value),
            ok_at_or_above: Number($("score-ok-at").value)
          },
          status_thresholds: {
            good: Number($("score-good").value),
            acceptable: Number($("score-acc").value)
          },
          risk_caps: {
            P0: Number($("score-p0").value),
            P1: Number($("score-p1").value)
          },
          attention_score_below: Number($("score-attention").value)
        };
      }
      function setStatus(text, isError) {
        var el = $("score-strictness-status");
        if (!el) return;
        el.textContent = text || "";
        el.style.color = isError ? "var(--danger, #b42318)" : "";
      }
      if ($("score-save")) {
        $("score-save").addEventListener("click", async function () {
          setStatus("Сохраняю…");
          try {
            var response = await request("/scoring-config", "/scoring-config", {
              method: "PUT",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(collectPatch())
            });
            if (!response.ok) throw new Error("Не удалось сохранить профиль");
            var data = await response.json();
            renderScoringStrictness(data);
            setStatus("Профиль сохранён. Запустите пересчёт периода или отложите на следующую загрузку.");
          } catch (error) {
            setStatus(error.message || String(error), true);
          }
        });
      }
      async function runRecompute(body) {
        setStatus("Запускаю пересчёт…");
        try {
          await request("/scoring-config", "/scoring-config", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(collectPatch())
          });
          var response = await request("/recompute", "/recompute", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
          });
          var data = await response.json().catch(function () { return {}; });
          if (!response.ok) throw new Error((data && data.detail) || "Пересчёт не запущен");
          renderScoringStrictness(data);
          if (data.scheduled) {
            setStatus("Отложено на следующую подгрузку данных.");
          } else {
            setStatus("Пересчёт витрины/зон запущен…");
            startScoringStrictnessPoll();
          }
        } catch (error) {
          setStatus(error.message || String(error), true);
        }
      }
      if ($("score-recompute-range")) {
        $("score-recompute-range").addEventListener("click", function () {
          runRecompute({
            apply_mode: "now",
            mode: "warehouse_zones",
            date_from: $("score-date-from").value,
            date_to: $("score-date-to").value
          });
        });
      }
      if ($("score-recompute-all")) {
        $("score-recompute-all").addEventListener("click", function () {
          if (!window.confirm("Пересчитать всю доступную витрину? Это может занять время.")) return;
          runRecompute({ apply_mode: "now", mode: "warehouse_zones", whole_range: true });
        });
      }
      if ($("score-deep-rescore-range")) {
        $("score-deep-rescore-range").addEventListener("click", function () {
          if (!window.confirm("Deep-пересчёт периода перепишет findings и может обновить overall в cases. Продолжить?")) return;
          runRecompute({
            apply_mode: "now",
            mode: "deep_rescore",
            date_from: $("score-date-from").value,
            date_to: $("score-date-to").value
          });
        });
      }
      if ($("score-next-load-days")) {
        $("score-next-load-days").addEventListener("click", function () {
          runRecompute({ apply_mode: "next_load", mode: "warehouse_zones" });
        });
      }
      if ($("score-next-load-range")) {
        $("score-next-load-range").addEventListener("click", function () {
          runRecompute({
            apply_mode: "next_load",
            mode: "warehouse_zones",
            date_from: $("score-date-from").value,
            date_to: $("score-date-to").value
          });
        });
      }
      if ($("score-next-load-all")) {
        $("score-next-load-all").addEventListener("click", function () {
          if (!window.confirm("На следующей загрузке пересчитать всю историю витрины?")) return;
          runRecompute({ apply_mode: "next_load", mode: "warehouse_zones", whole_range: true });
        });
      }
    }
    async function loadScoringStrictness() {
      var host = $("scoring-strictness");
      if (!host) return;
      try {
        var response = await request("/scoring-config", "/scoring-config");
        if (!response.ok) throw new Error("scoring-config unavailable");
        var data = await response.json();
        renderScoringStrictness(data);
        var job = data.recompute_job || {};
        if (job.status === "running" || job.status === "queued") startScoringStrictnessPoll();
      } catch (error) {
        host.innerHTML = '<p class="card-sub">Не удалось загрузить настройки жёсткости.</p>';
      }
    }
    async function loadSettingsPage() {
      var sessionHost = $("settings-session");
      var aboutHost = $("settings-about");
      if (sessionHost) {
        var sessionHtml = "";
        var appTok = MO.api.appSessionToken && MO.api.appSessionToken();
        if (appTok) {
          try {
            var statusResponse = await fetch("/api/methodist/account/status", { headers: headers() });
            var status = statusResponse.ok ? await statusResponse.json() : null;
            if (status && status.authenticated) {
              sessionHtml =
                '<div class="settings-session-row">' +
                "<div><b>" + esc(status.display_name || status.login || "Пользователь") + "</b>" +
                "<div class=\"card-sub\">Логин " + esc(status.login || "") +
                " · роль " + esc(roleLabelRu(status.role)) +
                " · доступ " + esc(accessLabelRu(status.mo_access)) +
                (status.reports_min_date ? " · отчёты с " + esc(status.reports_min_date) : "") +
                "</div></div>" +
                '<button class="button secondary" type="button" id="settings-logout">Выйти</button>' +
                "</div>";
            } else {
              sessionHtml = '<p class="card-sub">Сессия учётной записи не активна. Войдите снова.</p>';
            }
          } catch (error) {
            sessionHtml = '<p class="card-sub">Не удалось проверить сессию учётной записи.</p>';
          }
        } else if (token()) {
          sessionHtml =
            '<div class="settings-session-row">' +
            "<div><b>Вход по токену методиста</b>" +
            '<div class="card-sub">Полный доступ кабинета. Учётки создаются во вкладке «МО Аналитика» режима методиста.</div></div>' +
            '<a class="button secondary" href="/methodist/mis-kz">Учётные записи</a>' +
            "</div>";
        } else {
          sessionHtml = '<p class="card-sub">Нет активной сессии.</p>';
        }
        sessionHost.innerHTML = sessionHtml;
        var logoutBtn = $("settings-logout");
        if (logoutBtn) {
          logoutBtn.addEventListener("click", async function () {
            try {
              await fetch("/api/methodist/account/logout", { method: "POST", headers: headers() });
            } catch (error) {}
            if (MO.api.clearAppSessionToken) MO.api.clearAppSessionToken();
            setAuth(true, "Сессия завершена.");
          });
        }
      }
      if (aboutHost) {
        var versionText = "версия неизвестна";
        var freshText = "свежесть данных недоступна";
        try {
          var versionResponse = await fetch("/api/version", { headers: { Accept: "application/json" } });
          if (versionResponse.ok) {
            var version = await versionResponse.json();
            versionText = esc(version.version || "") +
              (version.git_commit ? " · " + esc(String(version.git_commit).slice(0, 12)) : "");
          }
        } catch (error) {}
        try {
          var freshResponse = await request("/freshness", "/freshness");
          if (freshResponse.ok) {
            var fresh = await freshResponse.json();
            freshText = fresh.latest_day
              ? ("последний день витрины " + esc(fresh.latest_day) +
                (fresh.lag_days != null ? " · отставание " + esc(fresh.lag_days) + " дн." : ""))
              : "день витрины ещё не зафиксирован";
          }
        } catch (error) {}
        aboutHost.innerHTML =
          "<p><b>Сборка:</b> " + versionText + "</p>" +
          "<p><b>Данные:</b> " + freshText + "</p>" +
          '<p class="card-sub">Справка открывается ссылкой внизу меню.</p>';
      }
      await loadScoringStrictness();
    }
    async function loadKpSync() {
      var kpis = $("kp-sync-kpis");
      var periodKpis = $("kp-sync-period-kpis");
      var periodTable = $("kp-sync-period-table");
      var changed = $("kp-sync-changed");
      var superseded = $("kp-sync-superseded");
      var recent = $("kp-sync-recent");
      if (!kpis) return;
      var response = await request("/kp-sync?days=90", "/kp-sync");
      if (!response.ok) throw new Error("Не удалось загрузить сверку протоколов МЗ.");
      var data = await response.json();
      var status = data.status || "missing";
      var tone = status === "success" || status === "missing" ? "good" : "review";
      var posts = data.post_periods || {};
      var syncP = data.sync_periods || {};
      if ($("kp-sync-freshness")) {
        $("kp-sync-freshness").textContent = data.sync_day
          ? ("сверка " + data.sync_day)
          : (data.crawled_utc || "нет сверки");
      }
      kpis.innerHTML =
        kpi("На сайте", data.site_count, "уникальных файлов") +
        kpi("В корпусе", data.catalog_n || data.local_count, "карточек каталога") +
        kpi("За ночь", data.changed_n, (data.added_n || 0) + " новых · " + (data.updated_n || 0) + " обновлённых") +
        kpi("Сверка", data.sync_day || (status === "missing" ? "ещё не было" : status), data.crawled_utc || "");
      if (periodKpis) {
        periodKpis.innerHTML =
          kpi("Посты за 7 дн", posts.d7, "по дате постановления МЗ") +
          kpi("Посты за 30 дн", posts.d30, "по дате постановления МЗ") +
          kpi("Посты за 90 дн", posts.d90, "по дате постановления МЗ") +
          kpi("Посты с начала года", posts.ytd, "год постановления = текущий") +
          kpi("Посты 2026", posts.y2026, "год в названии файла") +
          kpi("К нам за 30 дн", (syncP.d30 && syncP.d30.changed) || 0,
            ((syncP.d30 && syncP.d30.nights) || 0) + " ночей сверки");
      }
      if (periodTable) {
        var periodDefs = [
          ["7 дней", "d7"],
          ["30 дней", "d30"],
          ["90 дней", "d90"],
          ["С начала года", "ytd"]
        ];
        periodTable.innerHTML =
          '<table class="data-table"><thead><tr><th>Период</th><th>Посты МЗ</th><th>К нам новых</th><th>К нам обновлённых</th><th>Ночей сверки</th></tr></thead><tbody>' +
          periodDefs.map(function (pair) {
            var bucket = syncP[pair[1]] || {};
            return "<tr><td>" + esc(pair[0]) + "</td><td>" + esc(posts[pair[1]] || 0) +
              "</td><td>" + esc(bucket.added || 0) + "</td><td>" + esc(bucket.updated || 0) +
              "</td><td>" + esc(bucket.nights || 0) + "</td></tr>";
          }).join("") + "</tbody></table>";
      }
      function fmtDate(iso, dotted) {
        if (dotted) return dotted;
        if (iso && iso.length >= 10) return iso.slice(8, 10) + "." + iso.slice(5, 7) + "." + iso.slice(0, 4);
        return "";
      }
      function rowsHtml(items, extraCols) {
        if (!items || !items.length) return '<div class="empty">Нет записей</div>';
        return '<table class="data-table"><thead><tr><th>Рубрика</th><th>Файл</th><th>Пост МЗ</th>' +
          (extraCols ? "<th>Сверка</th><th>Статус</th>" : "") +
          "</tr></thead><tbody>" +
          items.map(function (row) {
            var viewer = row.relative_path
              ? '<a href="/proto-viewer.html?path=' + encodeURIComponent(row.relative_path) + '" target="_blank" rel="noopener noreferrer">' + esc(row.filename || row.title || row.relative_path) + "</a>"
              : esc(row.filename || row.title || "");
            var rubric = row.slug_ru || row.slug || "";
            var post = fmtDate(row.post_date_iso, row.post_date);
            if (row.post_number) post = post ? (post + " №" + row.post_number) : ("№" + row.post_number);
            var cells = "<tr><td>" + esc(rubric) + "</td><td>" + viewer + "</td><td>" + esc(post || "нет даты") + "</td>";
            if (extraCols) {
              cells += "<td>" + esc(row.synced_on || "") + "</td><td>" + esc(row.action || "") + "</td>";
            }
            return cells + "</tr>";
          }).join("") + "</tbody></table>";
      }
      if (changed) changed.innerHTML = rowsHtml([].concat(data.added || [], data.updated || []), true);
      if (superseded) superseded.innerHTML = rowsHtml(data.superseded || [], true);
      if (recent) recent.innerHTML = rowsHtml(data.recent_posts || [], false);
      renderKpSyncCharts(data);
      kpis.setAttribute("data-tone", tone);
    }
    function renderKpSyncCharts(data) {
      var history = data.history || [];
      var years = (data.by_year || []).filter(function (row) { return row.year !== "нет даты"; });
      var months = data.by_month || [];
      var slugs = data.by_slug || [];
      var histHost = $("kp-sync-history-chart");
      var histTable = $("kp-sync-history-table");
      if (histHost) {
        if (!history.length) {
          histHost.innerHTML = '<p class="empty">Пока одна или ни одной ночной сверки. Ряд появится после нескольких ночей.</p>';
        } else {
          MO.moChart(histHost, {
            tooltip: { trigger: "axis" },
            legend: { data: ["Новые", "Обновлённые"] },
            xAxis: { type: "category", data: history.map(function (row) { return row.date; }) },
            yAxis: { type: "value", minInterval: 1 },
            series: [
              { name: "Новые", type: "bar", stack: "in", barMaxWidth: 28, data: history.map(function (row) { return row.added; }) },
              { name: "Обновлённые", type: "bar", stack: "in", barMaxWidth: 28, data: history.map(function (row) { return row.updated; }) }
            ]
          }, { label: "Поступления протоколов по ночам сверки" });
        }
      }
      if (histTable) {
        if (!history.length) {
          histTable.innerHTML = "";
        } else {
          histTable.innerHTML =
            '<table class="data-table"><thead><tr><th>Дата сверки</th><th>Новые</th><th>Обновлённые</th><th>Заменены</th><th>На сайте</th><th>У нас</th></tr></thead><tbody>' +
            history.slice().reverse().map(function (row) {
              return "<tr><td>" + esc(row.date || "") + "</td><td>" + esc(row.added || 0) +
                "</td><td>" + esc(row.updated || 0) + "</td><td>" + esc(row.superseded || 0) +
                "</td><td>" + esc(row.site_count || 0) + "</td><td>" + esc(row.local_count || 0) + "</td></tr>";
            }).join("") + "</tbody></table>";
        }
      }
      var monthHost = $("kp-sync-month-chart");
      if (monthHost) {
        if (!months.length) {
          monthHost.innerHTML = '<p class="empty">В каталоге нет дат постановлений</p>';
        } else {
          MO.moChart(monthHost, {
            tooltip: { trigger: "axis" },
            xAxis: { type: "category", data: months.map(function (row) { return row.month; }) },
            yAxis: { type: "value", minInterval: 1 },
            series: [{ name: "Протоколы", type: "bar", barMaxWidth: 18, data: months.map(function (row) { return row.n; }) }]
          }, { label: "Число протоколов по месяцу постановления МЗ" });
        }
      }
      var yearHost = $("kp-sync-year-chart");
      if (yearHost) {
        if (!years.length) {
          yearHost.innerHTML = '<p class="empty">В каталоге нет дат постановлений</p>';
        } else {
          MO.moChart(yearHost, {
            tooltip: { trigger: "axis" },
            xAxis: { type: "category", data: years.map(function (row) { return row.year; }) },
            yAxis: { type: "value", minInterval: 1 },
            series: [{ name: "Протоколы", type: "bar", barMaxWidth: 22, data: years.map(function (row) { return row.n; }) }]
          }, { label: "Число протоколов по году постановления МЗ" });
        }
      }
      var slugHost = $("kp-sync-slug-chart");
      if (slugHost && MO.moDonut) {
        MO.moDonut(slugHost, slugs.slice(0, 8).map(function (row) {
          return { name: row.label || row.slug, value: row.n };
        }), {
          label: "Состав корпуса по рубрикам",
          centerText: String(data.catalog_n || 0),
          centerSub: "в каталоге",
          emptyText: "Каталог пуст"
        });
      }
    }
    async function loadPage(page) {
      $("global-error").hidden = true;
      try {
        if (page === "overview") await loadOverview();
        else if (page === "yesterday") await loadYesterday();
        else if (page === "queue") await loadCases(true);
        else if (page === "documents") await loadCases(false);
        else if (page === "doctors") await loadDoctorsDimension();
        else if (page === "reports") {
          await loadReports();
          try { await loadAccessLog(); } catch (e) {}
          try { await loadDataQuality(); } catch (e) {}
        }
        else if (page === "kp-sync") await loadKpSync();
        else if (page === "settings") await loadSettingsPage();
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
      documents: [
        "Визит", "Пациент", "Дата", "Врач / специальность", "Филиал", "Диагноз",
        "Оформление", "Диагноз (зона)", "План", "Причина", "Статус",
        "Итог", "№55 / градация", "Полнота проверки", "Надёжность"
      ],
      queue: [
        "Выбор", "Приоритет", "Раздел", "Визит", "Пациент", "Дата", "Филиал",
        "Врач / специальность", "Диагноз", "Оформление", "Диагноз (зона)", "План",
        "№55 / градация", "Причина", "Ответственный", "Срок", "Статус", "МО"
      ]
    };
    var COLUMN_DEFAULTS = {
      documents: [true, true, true, true, true, true, true, true, true, true, true, false, true, false, false],
      queue: [true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true]
    };
    function ensureColumnState() {
      if (!state.columnVisible.documents.length || state.columnVisible.documents.length !== COLUMN_MAP.documents.length) {
        state.columnVisible.documents = COLUMN_DEFAULTS.documents.slice();
      }
      if (!state.columnVisible.queue.length || state.columnVisible.queue.length !== COLUMN_MAP.queue.length) {
        state.columnVisible.queue = COLUMN_DEFAULTS.queue.slice();
      }
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
        state.zoneFilter = "";
        state.zoneBandFilter = "";
        state.attentionOnly = false;
        state.kpStatus = "";
        state.historyTier = "";
        $("period").value = state.period; $("compare").value = state.compare;
        $("case-search").value = "";
        $("sort-by").value = "date";
        $("sort-dir").value = "desc";
        $("date-from").value = ""; $("date-to").value = "";
        $("date-from-wrap").hidden = true; $("date-to-wrap").hidden = true;
        applyScoreEligibleOnly(true, true);
        document.querySelectorAll(".filter-pop").forEach(renderFilter);
        $("filters-panel").open = false;
        filtersChanged();
        showToast("Фильтры сброшены: только клинические приёмы");
      });
      document.querySelectorAll("[data-zone-preset]").forEach(function (btn) {
        btn.addEventListener("click", function () {
          applyZonePreset(btn.getAttribute("data-zone-preset") || "");
        });
      });
      var doctorMetric = $("doctor-zone-metric");
      if (doctorMetric) {
        doctorMetric.querySelectorAll("[data-doctor-zone]").forEach(function (btn) {
          btn.addEventListener("click", function () {
            state.doctorZoneMetric = btn.getAttribute("data-doctor-zone") || "zone1";
            renderDoctorZoneChart(state.data.doctorItems || []);
          });
        });
      }
      if ($("score-eligible-only")) {
        $("score-eligible-only").checked = true;
        $("score-eligible-only").disabled = true;
      }
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
      var shadowOnlyBtn = $("queue-shadow-attention-only");
      if (shadowOnlyBtn) {
        shadowOnlyBtn.addEventListener("click", function () {
          state.shadowAttentionOnly = !state.shadowAttentionOnly;
          shadowOnlyBtn.classList.toggle("active", state.shadowAttentionOnly);
          showToast(
            state.shadowAttentionOnly
              ? "Фильтр: shadow плохо/критично (не официальная оценка)"
              : "Фильтр shadow снят"
          );
          filtersChanged();
        });
      }
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
      if ($("yesterday-flow-dimension")) {
        $("yesterday-flow-dimension").addEventListener("change", function () {
          if (state.data.daily) renderYesterdayFlow(state.data.daily, this.value);
        });
      }
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
      if ($("density")) {
        $("density").addEventListener("change", function () {
          try { localStorage.setItem(DENSITY_KEY, this.value); } catch (error) {}
          applyPreferences();
          showToast(this.value === "compact" ? "Компактная плотность включена" : "Комфортная плотность включена");
        });
      }
      if ($("sidebar-help")) {
        $("sidebar-help").addEventListener("click", function () {
          switchPage("settings");
        });
      }
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
      if ($("drawer-prev")) $("drawer-prev").addEventListener("click", function () {
        var ids = state.caseNavIds || [];
        var idx = ids.indexOf(state.openCaseId);
        if (idx > 0) openCase(ids[idx - 1], state.trigger);
      });
      if ($("drawer-next")) $("drawer-next").addEventListener("click", function () {
        var ids = state.caseNavIds || [];
        var idx = ids.indexOf(state.openCaseId);
        if (idx >= 0 && idx < ids.length - 1) openCase(ids[idx + 1], state.trigger);
      });
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
      if ($("token-submit")) {
        $("token-submit").addEventListener("click", function () {
          var value = $("token-input").value.trim();
          if (!value) { $("auth-error").textContent = "Введите токен."; return; }
          if (MO.api.clearAppSessionToken) MO.api.clearAppSessionToken();
          if (MO.api.setToken) MO.api.setToken(value);
          else {
            try { localStorage.setItem(TOKEN_KEY, value); sessionStorage.setItem(TOKEN_KEY, value); } catch (e) {}
          }
          setAuth(false); loadCapabilities(); loadPage(state.page);
        });
      }
      if ($("account-login-submit")) {
        $("account-login-submit").addEventListener("click", async function () {
          var login = ($("account-login-input") && $("account-login-input").value || "").trim();
          var password = ($("account-password-input") && $("account-password-input").value || "").trim();
          if (!login || !password) {
            $("auth-error").textContent = "Введите логин и пароль.";
            return;
          }
          $("auth-error").textContent = "";
          try {
            var response = await fetch("/api/methodist/account/login", {
              method: "POST",
              headers: { Accept: "application/json", "Content-Type": "application/json" },
              body: JSON.stringify({ login: login, password: password })
            });
            var data = await response.json().catch(function () { return {}; });
            if (!response.ok) {
              $("auth-error").textContent = data.detail || "Неверный логин или пароль.";
              return;
            }
            if (MO.api.clearToken) MO.api.clearToken();
            if (MO.api.setAppSessionToken) MO.api.setAppSessionToken(data.session_token || "");
            if ($("account-password-input")) $("account-password-input").value = "";
            setAuth(false);
            await loadCapabilities();
            switchPage(state.page || "yesterday", false);
          } catch (error) {
            $("auth-error").textContent = "Не удалось войти. Проверьте сеть и повторите.";
          }
        });
        if ($("account-password-input")) {
          $("account-password-input").addEventListener("keydown", function (event) {
            if (event.key === "Enter") $("account-login-submit").click();
          });
        }
      }
      if ($("expert-login-submit")) {
        $("expert-login-submit").addEventListener("click", async function () {
          var login = ($("expert-login-input") && $("expert-login-input").value || "").trim();
          var password = ($("expert-password-input") && $("expert-password-input").value || "").trim();
          if (!login || !password) {
            $("auth-error").textContent = "Введите логин и пароль.";
            return;
          }
          $("auth-error").textContent = "";
          try {
            var response = await fetch("/api/expert/login", {
              method: "POST",
              headers: { Accept: "application/json", "Content-Type": "application/json" },
              body: JSON.stringify({ login: login, password: password })
            });
            var data = await response.json().catch(function () { return {}; });
            if (!response.ok) {
              $("auth-error").textContent = data.detail || "Неверный логин или пароль.";
              return;
            }
            MO.api.setExpertToken(data.session_token || "");
            state.expertDisplayName = data.display_name || data.login || login;
            if ($("expert-password-input")) $("expert-password-input").value = "";
            setAuth(false);
            await loadCapabilities();
            switchPage(state.page || "yesterday", false);
          } catch (error) {
            $("auth-error").textContent = "Не удалось войти. Проверьте сеть и повторите.";
          }
        });
        if ($("expert-password-input")) {
          $("expert-password-input").addEventListener("keydown", function (event) {
            if (event.key === "Enter") $("expert-login-submit").click();
          });
        }
      }
      if ($("expert-logout")) {
        $("expert-logout").addEventListener("click", async function () {
          try {
            await fetch("/api/expert/logout", {
              method: "POST",
              headers: headers()
            });
          } catch (error) {}
          MO.api.clearExpertToken();
          state.expertDisplayName = "";
          setAuth(true, "Сессия завершена.");
        });
      }
      window.addEventListener("popstate", function () { readUrl(); renderChips(); switchPage(state.page, false); });
      renderSavedViews(); refreshSavedViews();
    }
    async function init() {
      readUrl(); applyPreferences(); bind();
      state.methodology = "v4";
      renderChips();
      renderAnalysisRail();
      ensureColumnState();
      if ($("columns-manager")) $("columns-manager").hidden = true;
      if (!hasSession()) setAuth(true);
      else {
        setAuth(false);
        if (MO.api.appSessionToken && MO.api.appSessionToken()) {
          try {
            var appStatusResponse = await fetch("/api/methodist/account/status", { headers: headers() });
            if (appStatusResponse.ok) {
              var appStatus = await appStatusResponse.json();
              if (!appStatus.authenticated) {
                MO.api.clearAppSessionToken();
                setAuth(true);
                return;
              }
            }
          } catch (error) {}
        } else if (isExpertMode()) {
          try {
            var statusResponse = await fetch("/api/expert/status", { headers: headers() });
            if (statusResponse.ok) {
              var status = await statusResponse.json();
              if (status.authenticated) state.expertDisplayName = status.display_name || status.login || "";
              else {
                MO.api.clearExpertToken();
                setAuth(true);
                return;
              }
            }
          } catch (error) {}
        }
        await loadCapabilities();
        switchPage(state.page, false);
      }
    }
    MO.app = Object.freeze({ init: init, switchPage: switchPage, showToast: showToast });
    init();
  })(window.MO = window.MO || {});
