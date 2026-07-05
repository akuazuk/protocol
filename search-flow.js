/**
 * Protocol search-flow UI: macro stepper, context rail, session, analytics.
 */
(function (global) {
  "use strict";

  var SESSION_KEY = "protocol_search_flow_v1";
  var EVENTS_KEY = "protocol_search_flow_events";
  var RESTORE_DISMISS_KEY = "protocol_search_flow_restore_dismiss";

  var MACRO_PHASES = [
    { id: "query", label: "Запрос", jumpStep: "query" },
    { id: "refine", label: "Уточнение", jumpStep: "population" },
    { id: "protocols", label: "Протокол", jumpStep: "protocols" },
    { id: "clinical", label: "На приёме", jumpStep: "brief" },
  ];

  var SUB_STEPS = [
    { id: "brief", label: "Сводка" },
    { id: "condition", label: "Нозология" },
    { id: "section", label: "Раздел" },
    { id: "excerpt", label: "Цитата" },
  ];

  var POP_LABELS = {
    adult: "Взрослые",
    pediatric: "Дети",
    pregnant: "Беременные",
    emergency: "Неотложно",
    skipped: "Без уточнения",
  };

  var hooks = {};
  var historyPushing = false;

  function esc(s) {
    if (hooks.escapeHtml) return hooks.escapeHtml(s);
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function escAttr(s) {
    if (hooks.escapeHtmlAttr) return hooks.escapeHtmlAttr(s);
    return esc(s).replace(/"/g, "&quot;");
  }

  function phaseForStep(step) {
    if (!step || step === "query") return "query";
    if (step === "population" || step === "icd" || step === "rubric") return "refine";
    if (step === "protocols") return "protocols";
    return "clinical";
  }

  function phaseIndex(phaseId) {
    var i;
    for (i = 0; i < MACRO_PHASES.length; i++) {
      if (MACRO_PHASES[i].id === phaseId) return i;
    }
    return 0;
  }

  function subStepIndex(stepId) {
    if (stepId === "condition" || stepId === "section" || stepId === "excerpt") {
      var i;
      for (i = 0; i < SUB_STEPS.length; i++) {
        if (SUB_STEPS[i].id === stepId) return i;
      }
    }
    if (stepId === "brief" || stepId === "detail") return 0;
    return -1;
  }

  function trackEvent(name, detail) {
    detail = detail || {};
    try {
      var raw = sessionStorage.getItem(EVENTS_KEY);
      var list = raw ? JSON.parse(raw) : [];
      list.push({ t: Date.now(), name: name, detail: detail });
      if (list.length > 200) list = list.slice(-200);
      sessionStorage.setItem(EVENTS_KEY, JSON.stringify(list));
    } catch (e) {}
    if (global.console && console.debug) {
      console.debug("[search-flow]", name, detail);
    }
  }

  function saveSession(step) {
    try {
      var wiz = hooks.getSearchWizard ? hooks.getSearchWizard() : {};
      var ctx = hooks.getFunnelContext ? hooks.getFunnelContext() : {};
      var qEl = document.getElementById("q");
      sessionStorage.setItem(
        SESSION_KEY,
        JSON.stringify({
          step: step || wiz.step || "query",
          funnelContext: ctx,
          searchWizard: wiz,
          query: qEl ? qEl.value : "",
          ts: Date.now(),
        })
      );
    } catch (e) {}
  }

  function loadSession() {
    try {
      var raw = sessionStorage.getItem(SESSION_KEY);
      if (!raw) return null;
      var data = JSON.parse(raw);
      if (!data || !data.ts || Date.now() - data.ts > 86400000) return null;
      return data;
    } catch (e) {
      return null;
    }
  }

  function clearSession() {
    try {
      sessionStorage.removeItem(SESSION_KEY);
    } catch (e) {}
  }

  function renderMacroStepper(activeStep) {
    var el = document.getElementById("search-macro-stepper");
    if (!el) return;
    var activePhase = phaseForStep(activeStep);
    var activeIdx = phaseIndex(activePhase);
    var html = "";
    var pi;
    for (pi = 0; pi < MACRO_PHASES.length; pi++) {
      var ph = MACRO_PHASES[pi];
      var cls = "search-macro-stepper__item";
      if (pi < activeIdx) cls += " is-done";
      else if (pi === activeIdx) cls += " is-active";
      var canJump = pi < activeIdx && hooks.onJump;
      var innerTag = canJump ? "button" : "span";
      var innerCls = canJump ? "search-macro-stepper__btn" : "search-macro-stepper__label";
      var attrs = ' class="' + innerCls + '"';
      if (canJump) {
        attrs +=
          ' type="button" data-search-flow-jump="' +
          escAttr(ph.jumpStep) +
          '" title="Вернуться: ' +
          escAttr(ph.label) +
          '"';
      }
      html +=
        '<li class="' +
        cls +
        '"><' +
        innerTag +
        attrs +
        '><span class="search-macro-stepper__num">' +
        String(pi + 1) +
        '</span><span class="search-macro-stepper__text">' +
        esc(ph.label) +
        "</span></" +
        innerTag +
        "></li>";
    }
    el.innerHTML = html;
  }

  function populationLabel(ctx) {
    var pop = ctx.population || "";
    return POP_LABELS[pop] || "";
  }

  function renderContextRail(activeStep) {
    var el = document.getElementById("search-context-rail");
    if (!el) return;
    var ctx = hooks.getFunnelContext ? hooks.getFunnelContext() : {};
    var wiz = hooks.getSearchWizard ? hooks.getSearchWizard() : {};
    var rows = [];
    var q =
      hooks.stripPopulationContextFromQuery && document.getElementById("q")
        ? hooks.stripPopulationContextFromQuery(document.getElementById("q").value || "")
        : "";
    if (q) {
      var qShort = q.length > 80 ? q.slice(0, 77) + "…" : q;
      rows.push({ key: "Запрос", val: qShort, jump: "query" });
    }
    var popLbl = populationLabel(ctx);
    if (popLbl && activeStep !== "population") {
      rows.push({ key: "Аудитория", val: popLbl, jump: "population" });
    }
    if (ctx.icd_codes && ctx.icd_codes.length && activeStep !== "icd") {
      rows.push({
        key: "МКБ",
        val: ctx.icd_codes.slice(0, 4).join(", "),
        jump: "icd",
      });
    }
    if (ctx.rubric_slugs && ctx.rubric_slugs.length && activeStep !== "rubric") {
      rows.push({
        key: "Рубрика",
        val: ctx.rubric_slugs.slice(0, 2).join(", "),
        jump: "rubric",
      });
    }
    if (wiz.selectedProto || ctx.protocol_path) {
      var path = wiz.selectedProto || ctx.protocol_path;
      var title = hooks.formatProtocolDisplayTitle
        ? hooks.formatProtocolDisplayTitle(path, "")
        : path;
      if (title) {
        var tShort = title.length > 64 ? title.slice(0, 61) + "…" : title;
        rows.push({ key: "Протокол", val: tShort, jump: "protocols" });
      }
    }
    if (!rows.length || activeStep === "query") {
      el.hidden = true;
      el.innerHTML = "";
      return;
    }
    el.hidden = false;
    el.classList.toggle(
      "search-context-rail--compact",
      window.matchMedia && window.matchMedia("(max-width: 640px)").matches
    );
    el.classList.toggle("search-context-rail--mobile-compact", true);
    var html = '<p class="search-context-rail__title">Ваш путь</p><ul class="search-context-rail__list">';
    var ri;
    for (ri = 0; ri < rows.length; ri++) {
      var row = rows[ri];
      html +=
        '<li class="search-context-rail__row"><span class="search-context-rail__key">' +
        esc(row.key) +
        '</span><span class="search-context-rail__val">' +
        esc(row.val) +
        "</span>";
      if (hooks.onJump && row.jump && activeStep !== row.jump) {
        html +=
          '<button type="button" class="search-context-rail__edit" data-search-flow-jump="' +
          escAttr(row.jump) +
          '">изменить</button>';
      }
      html += "</li>";
    }
    html += "</ul>";
    el.innerHTML = html;
  }

  function renderSubStepper(activeStep) {
    var el = document.getElementById("search-sub-stepper");
    if (!el) return;
    if (phaseForStep(activeStep) !== "clinical") {
      el.hidden = true;
      el.innerHTML = "";
      return;
    }
    var normalized = activeStep;
    if (normalized === "detail") normalized = "brief";
    var activeIdx = subStepIndex(normalized);
    if (activeIdx < 0) {
      el.hidden = true;
      return;
    }
    el.hidden = false;
    var html = "";
    var si;
    for (si = 0; si < SUB_STEPS.length; si++) {
      var st = SUB_STEPS[si];
      var cls = "search-sub-stepper__item";
      if (si < activeIdx) cls += " is-done";
      else if (si === activeIdx) cls += " is-active";
      var canJump = si < activeIdx && hooks.onJump;
      var tag = canJump ? "button" : "span";
      var tagCls = canJump ? "search-sub-stepper__btn" : "search-sub-stepper__label";
      var attrs = ' class="' + tagCls + '"';
      if (canJump) {
        attrs += ' type="button" data-search-flow-jump="' + escAttr(st.id) + '"';
      }
      html +=
        '<li class="' +
        cls +
        '"><' +
        tag +
        attrs +
        ">" +
        esc(st.label) +
        "</" +
        tag +
        "></li>";
    }
    el.innerHTML = html;
  }

  function pushHistory(step) {
    if (!step || step === "query" || historyPushing) return;
    try {
      var state = { searchFlowStep: step, ts: Date.now() };
      history.pushState(state, "", "#search-step-" + step);
    } catch (e) {}
  }

  function afterWizardBarUpdate(step) {
    var shell = document.getElementById("search-flow-shell");
    if (shell) {
      shell.hidden = !step || step === "query";
    }
    if (!step || step === "query") {
      renderMacroStepper("query");
      renderContextRail("query");
      renderSubStepper("query");
      return;
    }
    renderMacroStepper(step);
    renderContextRail(step);
    renderSubStepper(step);
    saveSession(step);
    trackEvent("search_step_view", { step: step, phase: phaseForStep(step) });
    pushHistory(step);
  }

  function wireShell() {
    var shell = document.getElementById("search-flow-shell");
    if (!shell || shell.getAttribute("data-flow-wired")) return;
    shell.setAttribute("data-flow-wired", "1");
    shell.addEventListener("click", function (ev) {
      var btn =
        ev.target && ev.target.closest
          ? ev.target.closest("[data-search-flow-jump]")
          : null;
      if (!btn || !shell.contains(btn)) return;
      ev.preventDefault();
      var step = btn.getAttribute("data-search-flow-jump") || "";
      if (!step || !hooks.onJump) return;
      trackEvent("search_context_edit", { step: step });
      hooks.onJump(step);
    });
  }

  function initSettingsDrawer() {
    var trigger = document.getElementById("btn-search-settings");
    var drawer = document.getElementById("search-settings-drawer");
    var closeBtn = document.getElementById("btn-search-settings-close");
    if (!trigger || !drawer) return;
    var tierPanel = document.getElementById("search-tier-panel");
    var modeToolbar = document.querySelector(".search-mode-toolbar");
    var drawerBody = document.getElementById("search-settings-drawer-body");
    if (drawerBody) {
      if (modeToolbar && modeToolbar.parentNode !== drawerBody) {
        drawerBody.appendChild(modeToolbar);
      }
      if (tierPanel && tierPanel.parentNode !== drawerBody) {
        drawerBody.appendChild(tierPanel);
      }
    }
    trigger.addEventListener("click", function () {
      drawer.hidden = !drawer.hidden;
    });
    if (closeBtn) {
      closeBtn.addEventListener("click", function () {
        drawer.hidden = true;
      });
    }
  }

  function initInlinePopulation() {
    var wrap = document.getElementById("search-inline-population");
    if (!wrap || wrap.getAttribute("data-wired")) return;
    wrap.setAttribute("data-wired", "1");
    wrap.addEventListener("click", function (ev) {
      var btn =
        ev.target && ev.target.closest
          ? ev.target.closest("[data-inline-pop]")
          : null;
      if (!btn) return;
      var pop = btn.getAttribute("data-inline-pop") || "";
      if (!pop) return;
      global.__funnelContext = global.__funnelContext || {};
      global.__funnelContext.population = pop;
      global.__funnelContext.populationConfirmed = true;
      wrap.querySelectorAll("[data-inline-pop]").forEach(function (b) {
        b.classList.toggle("is-selected", b.getAttribute("data-inline-pop") === pop);
      });
      trackEvent("search_inline_population", { population: pop });
    });
  }

  function showRestoreBanner() {
    var banner = document.getElementById("search-flow-restore-banner");
    if (!banner) return;
    try {
      if (sessionStorage.getItem(RESTORE_DISMISS_KEY) === "1") return;
    } catch (e) {}
    var saved = loadSession();
    if (!saved || !saved.step || saved.step === "query") {
      banner.hidden = true;
      return;
    }
    banner.hidden = false;
    var resumeBtn = document.getElementById("search-flow-restore-resume");
    var dismissBtn = document.getElementById("search-flow-restore-dismiss");
    if (resumeBtn && !resumeBtn.getAttribute("data-wired")) {
      resumeBtn.setAttribute("data-wired", "1");
      resumeBtn.addEventListener("click", function () {
        try {
          if (saved.funnelContext) global.__funnelContext = saved.funnelContext;
          if (saved.searchWizard) global.__searchWizard = saved.searchWizard;
          if (saved.query) {
            var qEl = document.getElementById("q");
            if (qEl) qEl.value = saved.query;
          }
        } catch (e) {}
        banner.hidden = true;
        trackEvent("search_session_restore", { step: saved.step });
        if (hooks.onJump) hooks.onJump(saved.step);
      });
    }
    if (dismissBtn && !dismissBtn.getAttribute("data-wired")) {
      dismissBtn.setAttribute("data-wired", "1");
      dismissBtn.addEventListener("click", function () {
        banner.hidden = true;
        clearSession();
        try {
          sessionStorage.setItem(RESTORE_DISMISS_KEY, "1");
        } catch (e) {}
      });
    }
  }

  function initHistory() {
    global.addEventListener("popstate", function (ev) {
      var step =
        (ev.state && ev.state.searchFlowStep) ||
        (location.hash || "").replace("#search-step-", "");
      if (!step || !hooks.onBack) return;
      historyPushing = true;
      trackEvent("search_step_back", { step: step });
      if (hooks.onJump && step !== "query") {
        hooks.onJump(step);
      } else {
        hooks.onBack();
      }
      setTimeout(function () {
        historyPushing = false;
      }, 0);
    });
  }

  var STEP_TITLES = {
    population: {
      title: "Для кого подбираем протокол?",
      hint: "Детские и взрослые клинические протоколы различаются.",
    },
    icd: {
      title: "Выберите код МКБ-10",
      hint: "По коду диагноза ищем протокол в каталоге Минздрава РБ.",
    },
    rubric: {
      title: "Раздел каталога",
      hint: "Можно выбрать рубрику или искать по всем разделам.",
    },
  };

  function enhanceWizardCard(cardEl, stepId) {
    if (!cardEl || cardEl.getAttribute("data-flow-enhanced")) return;
    cardEl.setAttribute("data-flow-enhanced", "1");
    var meta = STEP_TITLES[stepId] || null;
    var lead = cardEl.querySelector(".search-wizard-icd-lead");
    if (lead && meta) {
      var header = document.createElement("div");
      header.className = "search-step-card__header";
      header.innerHTML =
        '<h3 class="search-step-card__title">' +
        esc(meta.title) +
        '</h3><p class="search-step-card__hint">' +
        esc(meta.hint) +
        "</p>";
      lead.replaceWith(header);
    } else if (lead) {
      lead.classList.add("search-step-card__hint");
    }
    var chips = cardEl.querySelector(".search-wizard-chips");
    if (chips) chips.classList.add("search-choice-list");
    cardEl.querySelectorAll(".search-wizard-chip").forEach(function (chip) {
      chip.classList.add("search-choice-card");
    });
    var actions = cardEl.querySelector(".search-wizard-step-actions");
    if (actions && !actions.querySelector(".btn-search-flow-back") && !actions.querySelector(".btn-search-wizard-back-inline")) {
      actions.classList.add("search-step-footer");
      var back = document.createElement("button");
      back.type = "button";
      back.className = "btn-search-flow-back";
      back.textContent = "← Назад";
      back.addEventListener("click", function () {
        trackEvent("search_step_back", { source: "footer" });
        if (hooks.onBack) hooks.onBack();
      });
      actions.insertBefore(back, actions.firstChild);
    }
  }

  function integrate(options) {
    hooks = options || {};
    wireShell();
    initSettingsDrawer();
    initInlinePopulation();
    initHistory();
    showRestoreBanner();
    var q = document.getElementById("q");
    if (q) q.classList.add("search-query--flow");
    var bar = document.getElementById("search-wizard-bar");
    if (bar) bar.classList.add("search-wizard-bar--flow");
  }

  global.ProtocolSearchFlow = {
    integrate: integrate,
    afterWizardBarUpdate: afterWizardBarUpdate,
    enhanceWizardCard: enhanceWizardCard,
    trackEvent: trackEvent,
    saveSession: saveSession,
    clearSession: clearSession,
    phaseForStep: phaseForStep,
    MACRO_PHASES: MACRO_PHASES,
  };
})(typeof window !== "undefined" ? window : globalThis);
