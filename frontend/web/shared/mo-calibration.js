(function (MO) {
  "use strict";

  var state = {
    items: [],
    caseIndex: 0,
    endpoint: "",
    dirty: false,
    saving: false
  };

  var labels = {
    complaints: "Жалобы",
    anamnesis: "Анамнез",
    objective_status: "Объективный статус",
    exam_data: "Данные обследований",
    exam_recommendations: "Рекомендации по обследованию",
    treatment_recommendations: "Рекомендации по лечению",
    follow_up: "Наблюдение",
    text: "Диагноз",
    icd: "МКБ"
  };

  function element(id) {
    return document.getElementById(id);
  }

  function complete(label) {
    return !!(label && label.verdict && label.score_pct !== null &&
      label.score_pct !== undefined && label.confidence !== null &&
      label.confidence !== undefined && String(label.rationale || "").trim().length >= 10);
  }

  function endpointName(endpoint) {
    return endpoint === "dx" ? "Диагноз по клиническим данным" : "План при принятом диагнозе";
  }

  function field(container, title, value, wide) {
    var article = document.createElement("article");
    article.className = "clinical-field" + (wide ? " wide" : "");
    var heading = document.createElement("h3");
    heading.textContent = title;
    var text = document.createElement("p");
    text.textContent = String(value || "Не указано");
    article.appendChild(heading);
    article.appendChild(text);
    container.appendChild(article);
  }

  function renderObject(container, value, prefix) {
    if (!value || typeof value !== "object") return;
    Object.keys(value).forEach(function (key) {
      var child = value[key];
      if (child === null || child === "" || child === undefined) return;
      var title = labels[key] || (prefix ? prefix + " · " + key : key);
      if (Array.isArray(child)) {
        field(container, title, child.map(function (item) {
          return typeof item === "object" ? JSON.stringify(item) : String(item);
        }).join("\n"), true);
      } else if (typeof child === "object") {
        renderObject(container, child, title);
      } else {
        field(container, title, child, String(child).length > 120);
      }
    });
  }

  function itemProgress(item) {
    var endpoints = item.required_endpoints || [];
    var done = endpoints.filter(function (endpoint) {
      return complete((item.labels || {})[endpoint]);
    }).length;
    return { done: done, total: endpoints.length };
  }

  function allProgress() {
    var done = 0;
    var total = 0;
    state.items.forEach(function (item) {
      var progress = itemProgress(item);
      done += progress.done;
      total += progress.total;
    });
    return { done: done, total: total };
  }

  function renderList() {
    var list = element("case-list");
    list.replaceChildren();
    state.items.forEach(function (item, index) {
      var progress = itemProgress(item);
      var button = document.createElement("button");
      button.type = "button";
      button.className = "calibration-case-button";
      button.setAttribute("aria-current", index === state.caseIndex ? "true" : "false");
      var title = document.createElement("strong");
      title.textContent = item.sample_id;
      var count = document.createElement("span");
      count.className = "case-progress";
      count.textContent = progress.done + "/" + progress.total;
      var meta = document.createElement("small");
      var specialty = (((item.clinical_case || {}).meta || {}).specialty || "Специальность не указана");
      meta.textContent = specialty;
      button.appendChild(title);
      button.appendChild(count);
      button.appendChild(meta);
      button.addEventListener("click", function () {
        selectCase(index);
      });
      list.appendChild(button);
    });
    var progress = allProgress();
    element("sidebar-progress").textContent = "Заполнено " + progress.done + " из " + progress.total;
    element("case-count").textContent = progress.done + "/" + progress.total;
  }

  function renderTabs(item) {
    var tabs = element("endpoint-tabs");
    tabs.replaceChildren();
    (item.required_endpoints || []).forEach(function (endpoint) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "button secondary endpoint-tab";
      button.setAttribute("role", "tab");
      button.setAttribute("aria-selected", endpoint === state.endpoint ? "true" : "false");
      button.textContent = endpointName(endpoint) +
        (complete((item.labels || {})[endpoint]) ? " · заполнено" : "");
      button.addEventListener("click", function () {
        selectEndpoint(endpoint);
      });
      tabs.appendChild(button);
    });
  }

  function populateForm(item) {
    var endpoint = state.endpoint;
    var saved = (item.labels || {})[endpoint] || {};
    element("label-verdict").value = saved.verdict || "";
    element("label-score").value = saved.score_pct === null || saved.score_pct === undefined ? "" : saved.score_pct;
    element("label-confidence").value = saved.confidence === null || saved.confidence === undefined ? "" : saved.confidence;
    element("label-harm").checked = saved.potential_harm === true;
    element("label-rationale").value = saved.rationale || "";
    var isDx = endpoint === "dx";
    element("icd-fit-wrap").hidden = !isDx;
    element("label-icd-fit").required = isDx;
    element("label-icd-fit").value = isDx ? (saved.icd_fit || "") : "na";
    element("review-stamp").textContent = saved.reviewed_at ?
      "Сохранено: " + saved.reviewed_at + " · " + (saved.reviewer_id || "методист") : "Оценка ещё не сохранена";
    state.dirty = false;
  }

  function renderCase() {
    var item = state.items[state.caseIndex];
    if (!item) return;
    var clinical = item.clinical_case || {};
    var meta = clinical.meta || {};
    element("case-title").textContent = "Случай " + item.sample_id;
    element("case-meta").textContent = [
      meta.specialty || "",
      meta.age_years ? "возраст " + meta.age_years : "",
      meta.sex || "",
      meta.visit_type || ""
    ].filter(Boolean).join(" · ");
    var progress = itemProgress(item);
    element("case-status").textContent = progress.done === progress.total ? "Заполнено" : progress.done + "/" + progress.total;

    var clinicalFields = element("clinical-fields");
    clinicalFields.replaceChildren();
    renderObject(clinicalFields, clinical.evidence || {}, "");
    renderObject(clinicalFields, clinical.diagnosis || {}, "");
    renderObject(clinicalFields, clinical.plan || {}, "");

    var protocol = item.protocol_context;
    element("protocol-card").hidden = !protocol;
    var protocolFields = element("protocol-context");
    protocolFields.replaceChildren();
    if (protocol) renderObject(protocolFields, protocol, "КП");

    if (!(item.required_endpoints || []).includes(state.endpoint)) {
      state.endpoint = item.required_endpoints[0] || "";
    }
    renderTabs(item);
    populateForm(item);
    renderList();
  }

  function confirmDiscard() {
    return !state.dirty || window.confirm("Несохранённая оценка будет потеряна. Продолжить?");
  }

  function selectCase(index) {
    if (index === state.caseIndex || !confirmDiscard()) return;
    state.caseIndex = index;
    state.endpoint = (state.items[index].required_endpoints || [])[0] || "";
    renderCase();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function selectEndpoint(endpoint) {
    if (endpoint === state.endpoint || !confirmDiscard()) return;
    state.endpoint = endpoint;
    renderCase();
  }

  function nextIncomplete() {
    if (!confirmDiscard()) return;
    var total = state.items.length;
    for (var offset = 0; offset < total; offset += 1) {
      var index = (state.caseIndex + offset) % total;
      var item = state.items[index];
      var endpoint = (item.required_endpoints || []).find(function (name) {
        return !complete((item.labels || {})[name]);
      });
      if (endpoint) {
        state.caseIndex = index;
        state.endpoint = endpoint;
        renderCase();
        return;
      }
    }
    element("save-state").textContent = "Все оценки заполнены";
  }

  async function parseResponse(response) {
    var data = {};
    try {
      data = await response.json();
    } catch (error) {}
    if (!response.ok) {
      throw new Error(data.detail || "Ошибка " + response.status);
    }
    return data;
  }

  async function load() {
    try {
      var response = await fetch("/api/methodist/mo/calibration/c6", {
        method: "GET",
        headers: MO.api.headers(),
        cache: "no-store",
        credentials: "same-origin"
      });
      var data = await parseResponse(response);
      state.items = data.items || [];
      if (!state.items.length) throw new Error("В review pack нет случаев.");
      state.endpoint = state.items[0].required_endpoints[0] || "";
      element("calibration-loading").hidden = true;
      element("calibration-workspace").hidden = false;
      element("save-state").textContent = data.status.passed ?
        "Разметка завершена" : "Доступ подтверждён";
      renderCase();
    } catch (error) {
      element("calibration-loading").hidden = true;
      var banner = element("calibration-error");
      banner.hidden = false;
      banner.textContent = String(error.message || error) +
        " Войдите через кабинет методиста и откройте страницу снова.";
      element("save-state").textContent = "Нет доступа";
    }
  }

  async function save(event) {
    event.preventDefault();
    if (state.saving) return;
    var item = state.items[state.caseIndex];
    var endpoint = state.endpoint;
    var currentLabel = (item.labels || {})[endpoint] || {};
    var payload = {
      verdict: element("label-verdict").value,
      score_pct: Number(element("label-score").value),
      potential_harm: element("label-harm").checked,
      icd_fit: endpoint === "dx" ? element("label-icd-fit").value : "na",
      confidence: Number(element("label-confidence").value),
      rationale: element("label-rationale").value.trim(),
      expected_reviewed_at: currentLabel.reviewed_at || ""
    };
    state.saving = true;
    element("save-label").disabled = true;
    element("save-state").textContent = "Сохраняем";
    try {
      var response = await fetch(
        "/api/methodist/mo/calibration/c6/labels/" +
          encodeURIComponent(item.sample_id) + "/" + encodeURIComponent(endpoint),
        {
          method: "PUT",
          headers: Object.assign({}, MO.api.headers(), { "Content-Type": "application/json" }),
          body: JSON.stringify(payload),
          cache: "no-store",
          credentials: "same-origin"
        }
      );
      var data = await parseResponse(response);
      item.labels[endpoint] = data.label;
      state.dirty = false;
      element("save-state").textContent = data.status.passed ?
        "Все 22 оценки сохранены" : "Сохранено · " + data.status.complete_label_n + "/" + data.status.expected_label_n;
      renderCase();
      if (!data.status.passed) nextIncomplete();
    } catch (error) {
      element("save-state").textContent = "Не сохранено: " + String(error.message || error);
    } finally {
      state.saving = false;
      element("save-label").disabled = false;
    }
  }

  element("label-form").addEventListener("input", function () {
    state.dirty = true;
    element("save-state").textContent = "Есть несохранённые изменения";
  });
  element("label-form").addEventListener("submit", save);
  element("next-incomplete").addEventListener("click", nextIncomplete);
  element("previous-case").addEventListener("click", function () {
    if (!confirmDiscard()) return;
    state.caseIndex = (state.caseIndex - 1 + state.items.length) % state.items.length;
    state.endpoint = (state.items[state.caseIndex].required_endpoints || [])[0] || "";
    renderCase();
  });
  window.addEventListener("beforeunload", function (event) {
    if (!state.dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });

  load();
})(window.MO = window.MO || {});
