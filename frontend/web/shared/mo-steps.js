(function (MO) {
  "use strict";

  var STEPS = [
    { id: 1, key: "patient", label: "Пациент" },
    { id: 2, key: "lens", label: "Линза" },
    { id: 3, key: "visit", label: "Визит" }
  ];
  var LENS_PAGE = 12;

  function enabled() {
    try {
      var params = new URLSearchParams(window.location.search || "");
      if (params.get("steps") === "0") return false;
    } catch (error) {}
    return true;
  }

  function currentStep() {
    try {
      var raw = parseInt(new URLSearchParams(window.location.search || "").get("step") || "3", 10);
      if (raw === 1 || raw === 2 || raw === 3) return raw;
    } catch (error) {}
    return 3;
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
      return "<li>" + esc(item.title_ru || item.code || "замечание") + "</li>";
    }).join("");
    return '<section class="case-stepper__panel" data-step-panel="3">' +
      "<h3>Визит</h3>" +
      "<p><strong>" + esc(gradeRu(record.overall_grade || record.status)) + "</strong> · " +
      esc(record.date || record.visit_date || "") + " · " + esc(record.specialization || "") + "</p>" +
      '<div class="case-stepper__chips">' + chips + "</div>" +
      '<p class="case-stepper__context">' + esc(summary.context || coverageLine(summary.coverage)) + "</p>" +
      '<p class="card-sub">' + esc(kpLine) + "</p>" +
      (defects ? "<ol>" + defects + "</ol>" : '<p class="empty">До 5 дефектов: сейчас пусто.</p>') +
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
    else slot.innerHTML = stepVisit(data, passport);
  }

  function bind(host, ctx) {
    if (!host) return;
    host.addEventListener("click", function (event) {
      var crumb = event.target.closest("[data-step]");
      if (crumb) {
        setStep(Number(crumb.getAttribute("data-step") || "3"));
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

  async function mount(host, data, caseId) {
    if (!host) return;
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
