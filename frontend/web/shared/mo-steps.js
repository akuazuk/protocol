(function (MO) {
  "use strict";

  var STEPS = [
    { id: 1, key: "patient", label: "Пациент" },
    { id: 2, key: "lens", label: "Линза" },
    { id: 3, key: "visit", label: "Визит" },
    { id: 4, key: "proof", label: "Доказательство" },
    { id: 5, key: "decision", label: "Решение" }
  ];
  var LENS_PAGE = 12;
  var QUOTE_MAX = 240;

  function enabled() {
    try {
      var params = new URLSearchParams(window.location.search || "");
      return params.get("steps") === "1";
    } catch (error) {}
    return false;
  }

  function currentStep() {
    try {
      var raw = parseInt(new URLSearchParams(window.location.search || "").get("step") || "3", 10);
      if (raw >= 1 && raw <= 5) return raw;
    } catch (error) {}
    return 3;
  }

  function currentProof() {
    try {
      return String(new URLSearchParams(window.location.search || "").get("proof") || "");
    } catch (error) {
      return "";
    }
  }

  function currentLens() {
    try {
      return String(new URLSearchParams(window.location.search || "").get("lens") || "all");
    } catch (error) {
      return "all";
    }
  }

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function gradeRu(value) {
    var map = {
      good: "Хорошо",
      fair: "С замечанием",
      poor: "Слабо",
      important: "Важно",
      critical: "Критично",
      na: "Нет оценки"
    };
    return map[String(value || "")] || "Нет оценки";
  }

  function bandRu(value) {
    var map = { ok: "норма", weak: "слабо", bad: "плохо", na: "нет данных" };
    return map[String(value || "")] || "нет данных";
  }

  function setStep(step, extra) {
    extra = extra || {};
    try {
      var url = new URL(window.location.href);
      url.searchParams.set("page", url.searchParams.get("page") || "case");
      url.searchParams.set("step", String(step));
      if (extra.lens) url.searchParams.set("lens", extra.lens);
      if (extra.date) url.searchParams.set("lab_date", extra.date);
      if (extra.proof) url.searchParams.set("proof", extra.proof);
      if (extra.clearProof) url.searchParams.delete("proof");
      window.history.replaceState(window.history.state, "", url);
    } catch (error) {}
  }

  function crumbsHtml(step) {
    return '<nav class="case-stepper__crumbs" aria-label="Шаги разбора">' +
      STEPS.map(function (item) {
        var current = item.id === step;
        return '<button type="button" class="case-stepper__crumb' +
          (current ? " is-current" : "") +
          '" data-step="' + item.id + '"' +
          (current ? ' aria-current="step"' : "") + ">" +
          item.id + " " + esc(item.label) + "</button>";
      }).join('<span class="case-stepper__sep" aria-hidden="true">→</span>') +
      "</nav>";
  }

  function coverageLine(coverage) {
    coverage = coverage || {};
    var visits = Number(coverage.n_visits || 0);
    var specs = Number(coverage.n_specialties || 0);
    var labs = Number(coverage.n_lab_dates || 0);
    if (!visits && !specs && !labs) return "Оцениваем только этот визит.";
    return visits + " визитов, " + specs + " специальностей, " + labs + " дней анализов";
  }

  function stepPatient(passport) {
    var coverage = (passport && passport.coverage) || {};
    var specs = (passport && passport.specialties) || [];
    var tiles = specs.map(function (item) {
      return '<li class="case-stepper__tile">' +
        "<strong>" + esc(item.specialty || "Специальность") + "</strong>" +
        "<span>" + Number(item.n_visits || 0) + " визитов · " +
        esc(item.last_date || "-") + " · " + esc(item.last_icd || "без МКБ") +
        " · " + esc(gradeRu(item.last_grade)) + "</span></li>";
    }).join("");
    return '<section class="case-stepper__panel" data-step-panel="1">' +
      "<h3>Пациент</h3>" +
      '<p class="case-stepper__context">' + esc(passport && passport.context || coverageLine(coverage)) + "</p>" +
      (tiles ? '<ul class="case-stepper__tiles">' + tiles + "</ul>" : '<p class="empty">Нет ключа - только этот визит.</p>') +
      "</section>";
  }

  function stepLens(passport, lens, labDate) {
    var visits = ((passport && passport.visits) || []).slice();
    if (lens && lens !== "all" && lens !== "labs") {
      visits = visits.filter(function (item) { return String(item.specialty || "") === lens; });
    }
    var shown = visits.slice(0, LENS_PAGE);
    var more = Math.max(0, visits.length - shown.length);
    var rows = shown.map(function (item) {
      return '<li><button type="button" class="case-stepper__row" data-step="3">' +
        esc(item.visit_date || "без даты") + " · " + esc(item.specialty || "спец.") +
        " · " + esc(item.diagnosis_code || "без МКБ") +
        " · " + esc(gradeRu(item.overall_grade)) + "</button></li>";
    }).join("");
    var dates = ((passport && passport.lab_dates) || []).map(function (day) {
      return '<button type="button" class="case-stepper__lab-day' +
        (day === labDate ? " is-current" : "") +
        '" data-lab-date="' + esc(day) + '">' + esc(day) + "</button>";
    }).join("");
    return '<section class="case-stepper__panel" data-step-panel="2">' +
      "<h3>Линза</h3>" +
      '<p class="case-stepper__context">Все визиты, одна специальность или анализы. На экране 12 строк.</p>' +
      '<div class="case-stepper__lenses">' +
      '<button type="button" data-lens="all">Все</button>' +
      '<button type="button" data-lens="labs">Анализы</button></div>' +
      (dates ? '<div class="case-stepper__lab-days">' + dates + "</div>" : "") +
      (lens === "labs" && !dates ? '<p class="empty">Дней анализов нет.</p>' : "") +
      (lens === "labs" ? "" : '<ul class="case-stepper__list">' + (rows || '<li class="empty">Нет визитов в линзе.</li>') + "</ul>") +
      (more ? '<p class="card-sub">Ещё ' + more + "</p>" : "") +
      "</section>";
  }

  function stepVisit(data, passport) {
    var record = (data && data.record) || {};
    var zones = (data && data.zones) || {};
    var findings = ((data && data.findings) || []).filter(function (item) {
      return item && !item.is_shadow;
    }).slice(0, 5);
    var summary = (data && data.passport_summary) || {};
    var kp = (data && data.protocol_suggest) || {};
    var kpLine = kp.available === false
      ? (kp.reason || "Протокол не подобран - не штрафуем")
      : ((kp.items && kp.items[0] && kp.items[0].title) || "Протокол: смотрим подбор");
    var chips = ["zone1", "zone2a", "zone2b"].map(function (key) {
      var label = key === "zone1" ? "Оформление" : key === "zone2a" ? "Диагноз" : "План";
      var band = zones[key + "_band"] || zones[key] && zones[key].band;
      return '<span class="case-stepper__chip">' + esc(label) + " · " + esc(bandRu(band)) + "</span>";
    }).join("");
    var defects = findings.map(function (item) {
      var code = item.code || item.finding_code || "";
      return '<li><button type="button" class="case-stepper__row" data-step="4" data-proof="' +
        esc(code) + '">' + esc(item.title_ru || code || "замечание") + "</button></li>";
    }).join("");
    return '<section class="case-stepper__panel" data-step-panel="3">' +
      "<h3>Визит</h3>" +
      "<p><strong>" + esc(gradeRu(record.overall_grade || record.status)) + "</strong> · " +
      esc(record.date || record.visit_date || "") + " · " + esc(record.specialization || "") + "</p>" +
      '<div class="case-stepper__chips">' + chips + "</div>" +
      '<p class="case-stepper__context">' + esc(summary.context || coverageLine(summary.coverage)) + "</p>" +
      '<p class="card-sub">' + esc(kpLine) + "</p>" +
      (defects ? "<ol>" + defects + "</ol>" : '<p class="empty">До 5 дефектов: сейчас пусто.</p>') +
      '<p><button type="button" class="button secondary" data-step="4">К доказательству</button> ' +
      '<button type="button" class="button" data-step="5">К решению</button></p>' +
      "</section>";
  }

  function officialFindings(data) {
    return ((data && data.findings) || []).filter(function (item) {
      return item && !item.is_shadow;
    });
  }

  function clipQuote(value) {
    var text = String(value || "").replace(/\s+/g, " ").trim();
    if (text.length <= QUOTE_MAX) return text;
    return text.slice(0, QUOTE_MAX - 1).replace(/\s+\S*$/, "").trim() + "...";
  }

  function proofBasket(item) {
    if (!item) return "контекст";
    if (item.is_shadow) return "черновик";
    var axis = String(item.axis || "");
    if (axis === "history" || axis === "context") return "контекст";
    return "дефект";
  }

  function stepProof(data) {
    var items = officialFindings(data);
    var proof = currentProof();
    var idx = 0;
    items.forEach(function (item, i) {
      if ((item.code || item.finding_code) === proof) idx = i;
    });
    var item = items[idx] || null;
    var assessment = (data && data.assessment) || {};
    var version = assessment.evaluator_version || assessment.scorer_version ||
      (data && data.record && data.record.evaluation_run_id) || "";
    var quote = item ? clipQuote(item.detail_ru || item.evidence || item.source_ref || item.title_ru) : "";
    var prev = items[idx - 1];
    var next = items[idx + 1];
    return '<section class="case-stepper__panel" data-step-panel="4">' +
      "<h3>Доказательство</h3>" +
      (item
        ? "<p><strong>" + esc(item.title_ru || item.code) + "</strong> · " +
          esc(proofBasket(item)) + "</p>" +
          '<blockquote class="case-stepper__quote">' + esc(quote || "Цитаты нет.") + "</blockquote>" +
          '<p class="card-sub">evaluator_version: ' + esc(version || "нет") + "</p>"
        : '<p class="empty">Нет официального замечания для доказательства.</p>') +
      '<div class="case-stepper__proof-nav">' +
      '<button type="button" data-step="4" data-proof="' + esc((prev && (prev.code || prev.finding_code)) || "") + '"' +
      (prev ? "" : " disabled") + ">Предыдущее</button>" +
      "<span>" + (items.length ? (idx + 1) + " / " + items.length : "0 / 0") + "</span>" +
      '<button type="button" data-step="4" data-proof="' + esc((next && (next.code || next.finding_code)) || "") + '"' +
      (next ? "" : " disabled") + ">Следующее</button>" +
      '<button type="button" class="button" data-step="5">К решению</button>' +
      "</div></section>";
  }

  function stepDecision() {
    return '<section class="case-stepper__panel" data-step-panel="5">' +
      "<h3>Решение</h3>" +
      '<p class="case-stepper__context">Три вердикта, комментарий и PDF. Статус CRM сюда не выносится.</p>' +
      "</section>";
  }

  function hostHtml() {
    return '<div id="case-stepper" class="case-stepper">' +
      crumbsHtml(currentStep()) +
      '<div id="case-step-slot" class="case-stepper__slot"></div></div>';
  }

  function renderSlot(host, data, passport) {
    if (!host) return;
    var step = currentStep();
    var crumbs = host.querySelector(".case-stepper__crumbs");
    if (crumbs) crumbs.outerHTML = crumbsHtml(step);
    var slot = host.querySelector("#case-step-slot");
    if (!slot) return;
    if (step === 1) slot.innerHTML = stepPatient(passport);
    else if (step === 2) slot.innerHTML = stepLens(passport, currentLens(), "");
    else if (step === 4) slot.innerHTML = stepProof(data);
    else if (step === 5) slot.innerHTML = stepDecision();
    else slot.innerHTML = stepVisit(data, passport);
    syncDecisionDock(host._decisionDock, step);
  }

  function syncDecisionDock(dock, step) {
    if (!dock) return;
    dock.hidden = step !== 5;
    var crm = dock.querySelector("#drawer-status");
    var crmLabel = crm && crm.closest("label");
    if (crmLabel) crmLabel.hidden = true;
  }

  function bind(host, ctx) {
    if (!host) return;
    host.addEventListener("click", function (event) {
      var crumb = event.target.closest("[data-step]");
      if (crumb && !crumb.disabled) {
        var nextStep = Number(crumb.getAttribute("data-step") || "3");
        var proof = crumb.getAttribute("data-proof") || "";
        setStep(nextStep, proof ? { proof: proof } : (nextStep === 4 ? {} : { clearProof: true }));
        renderSlot(host, ctx.data, ctx.passport);
        return;
      }
      var lens = event.target.closest("[data-lens]");
      if (lens) {
        setStep(2, { lens: lens.getAttribute("data-lens") });
        renderSlot(host, ctx.data, ctx.passport);
        return;
      }
      var day = event.target.closest("[data-lab-date]");
      if (day) {
        setStep(2, { lens: "labs", date: day.getAttribute("data-lab-date") });
        renderSlot(host, ctx.data, ctx.passport);
      }
    });
  }

  async function loadPassport(caseId) {
    if (!caseId || !MO.api || !MO.api.request) return { ok: false };
    try {
      var response = await MO.api.request("/cases/" + encodeURIComponent(caseId) + "/passport");
      if (!response || !response.ok) return { ok: false };
      return await response.json();
    } catch (error) {
      return { ok: false };
    }
  }

  async function mount(host, data, caseId, options) {
    if (!host) return;
    options = options || {};
    host._decisionDock = options.decisionDock || null;
    host.innerHTML = hostHtml();
    var passport = data && data.passport_summary && data.passport_summary.ok
      ? data.passport_summary
      : {};
    var full = await loadPassport(caseId);
    if (full && full.ok) {
      passport = full;
      passport.lab_dates = [];
    }
    if (full && full.ok) {
      try {
        var labs = await MO.api.request("/cases/" + encodeURIComponent(caseId) + "/passport/labs");
        if (labs && labs.ok) {
          var body = await labs.json();
          passport.lab_dates = body.dates || [];
        }
      } catch (error) {}
    }
    renderSlot(host, data, passport);
    bind(host, { data: data, passport: passport });
  }

  MO.steps = Object.freeze({
    enabled: enabled,
    currentStep: currentStep,
    hostHtml: hostHtml,
    mount: mount,
    setStep: setStep
  });
})(window.MO = window.MO || {});
