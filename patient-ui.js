(function () {
  "use strict";

  var HISTORY_KEY = "protocol_patient_history_v2";
  var CHECKLIST_KEY = "protocol_patient_checklist_v1";
  var ONBOARD_KEY = "protocol_patient_onboard_done";
  var REPORT_KEY = "protocol_patient_last_report_v3";
  var REMINDER_KEY = "protocol_patient_reminder_v1";
  var SESSION_KEY = "protocol_patient_session_token";
  var QUESTION_TONE_KEY = "protocol_patient_question_tone_v2";

  var questionTonesCatalog = [
    { id: "serious", label_ru: "Строго и серьёзно", emoji: "serious", icon: "serious", description_ru: "Коротко, по делу, без шуток", default: true, accent: "#1e3a5f" },
    { id: "official", label_ru: "Официально", emoji: "official", icon: "official", description_ru: "Деловой стиль, обращение на «Вы»", accent: "#1d4ed8" },
    { id: "playful", label_ru: "Шуточно", emoji: "playful", icon: "playful", description_ru: "По-домашнему, с лёгким юмором про поликлинику и выписку", accent: "#b8860b" },
  ];
  var selectedQuestionTone = "serious";

  function normalizeQuestionToneId(tone) {
    var map = { friendly: "serious", light: "playful", дружелюбно: "serious", "с лёгкостью": "playful" };
    var t = (tone || "").trim();
    return map[t] || t || "serious";
  }

  var params = new URLSearchParams(window.location.search);
  var clinicId = params.get("clinic") || "";
  var tierId = params.get("tier") || "";
  var noHistoryMode = params.get("no_history") === "1";
  var paidToken = params.get("paid") || localStorage.getItem("protocol_patient_payment_token") || "";
  var monetization = {
    monetization_enabled: false,
    payment_required: false,
    show_tier_picker: false,
    show_prices: true,
    default_tier_id: "basic",
    payment_note_ru: "",
    value_banner_ru: "",
    tiers: [],
  };

  var lastReport = null;
  var lastProtocolLinks = [];
  var reviewFingerprint = null;
  var clinicConfig = null;
  var selectedTier = tierId || "basic";
  var useSse = true;

  var kzFilesList = [];
  var labFilesList = [];
  var kzMaxFiles = 1;
  var kzCameraInput = document.getElementById("kz-files-camera");
  var kzPickInput = document.getElementById("kz-files-pick");
  var labCameraInput = document.getElementById("lab-files-camera");
  var labPickInput = document.getElementById("lab-files-pick");
  var consentEl = document.getElementById("consent");
  var btn = document.getElementById("btn-check");
  var statusEl = document.getElementById("status");
  var loader = document.getElementById("loader");
  var loaderText = document.getElementById("loader-text");
  var loaderSub = document.getElementById("loader-sub");
  var formCard = document.getElementById("form-card");
  var resultCard = document.getElementById("result-card");
  var onboard = document.getElementById("onboard");

  function escapeHtml(s) {
    return String(s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  var LUX_ICON_PATHS = {
    serious: '<circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2.5"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3"/>',
    official: '<path d="M7 4h10a1 1 0 0 1 1 1v14l-4-2.5L10 19l-4-2.5V5a1 1 0 0 1 1-1z"/><path d="M9.5 8h5M9.5 11.5h5M9.5 15h3"/>',
    playful: '<path d="M12 3l1.2 3.6L17 7.8l-3 2.2.9 3.5L12 11.8 9.1 13.5l.9-3.5-3-2.2 3.8-1.2z"/><path d="M5 5l.8 1.6L7.4 7l-1.6.8L5 9.4 4.2 7.8 2.6 7l1.6-.8z"/><path d="M18.5 14l.7 1.4 1.6.8-1.6.8-.7 1.4-.7-1.4-1.6-.8 1.6-.8z"/>',
    speech: '<path d="M6 8.5a5.5 5.5 0 0 1 9.3-3.9A4.5 4.5 0 0 1 12 17H8l-3.5 2.5V16.2A5.4 5.4 0 0 1 6 8.5z"/>',
    history: '<path d="M6 4.5v3H9"/><path d="M6 9a6 6 0 1 0 1.8 4.2"/><path d="M9 11.5h4M9 14.5h6"/>',
    stethoscope: '<path d="M6.5 5.5a3 3 0 0 1 6 0v5a2.5 2.5 0 0 0 5 0V8"/><circle cx="17.5" cy="8" r="2"/><path d="M17.5 10v2.5a4 4 0 0 1-8 0"/>',
    dna: '<path d="M8 4c3 0 4.5 2 4.5 4S11 12 8 12s-4.5 2-4.5 4 1.5 4 4.5 4"/><path d="M16 4c-3 0-4.5 2-4.5 4s1.5 4 4.5 4 4.5 2 4.5 4-1.5 4-4.5 4"/><path d="M8.5 7h7M8.5 17h7"/>',
    scan: '<rect x="4.5" y="5.5" width="15" height="13" rx="2"/><path d="M8 9.5h8M8 12.5h6M8 15.5h4"/><path d="M7 4.5v2M17 4.5v2M7 17.5v2M17 17.5v2"/>',
    pill: '<rect x="5" y="9" width="14" height="6" rx="3"/><path d="M12 9v6"/><path d="M5 12H3M21 12h-2"/>',
    calendar: '<rect x="4.5" y="6" width="15" height="13" rx="2"/><path d="M8 4.5v3M16 4.5v3M4.5 10h15"/><path d="M9 14h2M13 14h2"/>',
    lab: '<path d="M9 4.5h6l-2 5.5v7.5H11V10z"/><path d="M7.5 17.5h9"/><circle cx="14.5" cy="13" r="1"/>',
    protocol: '<path d="M6.5 4.5h9l3 3v12a1 1 0 0 1-1 1h-11a1 1 0 0 1-1-1v-14a1 1 0 0 1 1-1z"/><path d="M15.5 4.5v3h3"/><path d="M8.5 11h7M8.5 14.5h5"/>',
    document: '<path d="M7 4.5h7l3 3v12a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1v-14a1 1 0 0 1 1-1z"/><path d="M14 4.5v3h3"/><path d="M8.5 12h7M8.5 15.5h5"/>',
    chat: '<path d="M5 6.5a2.5 2.5 0 0 1 2.5-2.5h9A2.5 2.5 0 0 1 19 6.5v6A2.5 2.5 0 0 1 16.5 15H11l-3.5 2.5V15H7.5A2.5 2.5 0 0 1 5 12.5z"/>'
  };

  function luxIconHtml(id, extraClass) {
    var key = id || "chat";
    var paths = LUX_ICON_PATHS[key] || LUX_ICON_PATHS.chat;
    var cls = "lux-icon" + (extraClass ? " " + extraClass : "");
    return (
      '<span class="' + cls + '" aria-hidden="true">' +
      '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.55" stroke-linecap="round" stroke-linejoin="round">' +
      paths +
      "</svg></span>"
    );
  }

  function resolveProtocolLink(link, fallbackTitle, fallbackPath) {
    if (link && (link.nav_url || link.url || link.pdf_url || link.path)) return link;
    var path = (link && link.path) || fallbackPath || "";
    var title = (link && link.title) || fallbackTitle || "";
    var i;
    for (i = 0; i < lastProtocolLinks.length; i++) {
      var pl = lastProtocolLinks[i];
      if (path && pl.path === path) return pl;
      if (title && pl.title && pl.title.toLowerCase() === title.toLowerCase()) return pl;
    }
    if (lastProtocolLinks.length === 1 && (title || path)) return lastProtocolLinks[0];
    return link || null;
  }

  function protocolNavHref(path) {
    var p = String(path || "").trim();
    if (!p) return "";
    return "/proto-viewer.html?path=" + encodeURIComponent(p);
  }

  function renderProtocolLink(link, fallbackTitle, fallbackPath) {
    var resolved = resolveProtocolLink(link, fallbackTitle, fallbackPath);
    var path = (resolved && resolved.path) || fallbackPath || "";
    var href =
      (resolved && (resolved.nav_url || resolved.url)) ||
      protocolNavHref(path) ||
      (resolved && resolved.pdf_url) ||
      "";
    if (!href) {
      return fallbackTitle ? escapeHtml(fallbackTitle) : "";
    }
    var title = (resolved && resolved.title) || fallbackTitle || "Клинический протокол Минздрава";
    return (
      '<a class="proto-link" href="' +
      escapeHtml(href) +
      '" target="_blank" rel="noopener noreferrer" title="Открыть навигацию по протоколу">' +
      '<span class="proto-link__icon" aria-hidden="true">КП</span>' +
      '<span class="proto-link__text">' +
      escapeHtml(title) +
      "</span></a>"
    );
  }

  function renderProtocolChip(link) {
    var resolved = resolveProtocolLink(link);
    if (!resolved || !(resolved.nav_url || resolved.url || resolved.pdf_url || resolved.path)) return "";
    var rubric = resolved.rubric ? '<span class="proto-chip__rubric">' + escapeHtml(resolved.rubric) + "</span>" : "";
    return (
      '<li class="proto-chip">' +
      '<span class="proto-chip__icon" aria-hidden="true">КП</span>' +
      '<div class="proto-chip__body">' +
      renderProtocolLink(resolved) +
      rubric +
      "</div></li>"
    );
  }

  function renderProtocolStrip(links) {
    var strip = document.getElementById("protocol-strip");
    if (!strip) return;
    if (!links || !links.length) {
      strip.classList.add("hidden");
      strip.innerHTML = "";
      return;
    }
    strip.classList.remove("hidden");
    var primary = links[0];
    var extra = links.length > 1 ? links.length - 1 : 0;
    var html =
      '<div class="protocol-strip__label">Клинический протокол Минздрава (ориентир для сверки)</div>' +
      '<ul class="proto-chips">' +
      renderProtocolChip(primary);
    if (extra > 0) {
      html +=
        '<li class="proto-chip proto-chip--more"><span class="proto-chip__more">+ ещё ' +
        extra +
        " протокол(ов) в системе - основной указан выше</span></li>";
    }
    html += "</ul>";
    strip.innerHTML = html;
  }

  function setAgainButtonVisible(visible, label) {
    var btn = document.getElementById("btn-again");
    var bar = document.getElementById("top-bar");
    if (btn) {
      if (label) {
        btn.textContent = label;
      } else if (!visible || btn.textContent === "Загрузить правильный документ") {
        btn.textContent = "Проверить другой документ";
      }
      btn.classList.toggle("hidden", !visible);
      btn.hidden = !visible;
    }
    if (bar) bar.classList.toggle("top-bar--result", !!visible);
  }

  function renderProtocolLinksList(links, limit) {
    if (!links || !links.length) return "";
    var n = limit || links.length;
    var parts = [];
    var i;
    for (i = 0; i < links.length && i < n; i++) {
      parts.push(renderProtocolLink(links[i]));
    }
    return parts.join("");
  }

  function track(event, meta) {
    try {
      fetch(window.location.origin + "/api/patient/analytics", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          event: event,
          clinic_id: clinicId || null,
          tier_id: selectedTier,
          text_hash: reviewFingerprint || null,
          meta: meta || {},
        }),
      }).catch(function () {});
    } catch (e) {}
  }

  function fileKey(f) {
    return [f.name, f.size, f.lastModified].join("|");
  }

  function mergeFilesIntoList(list, incoming, maxFiles) {
    if (maxFiles === 1 && incoming && incoming.length) {
      return [incoming[0]];
    }
    var map = {};
    var order = [];
    function add(f) {
      var k = fileKey(f);
      if (!map[k]) {
        map[k] = f;
        order.push(k);
      }
    }
    var i;
    for (i = 0; i < list.length; i++) add(list[i]);
    for (i = 0; i < incoming.length; i++) add(incoming[i]);
    return order.map(function (k) { return map[k]; });
  }

  function renderFileChips(list, containerId, dropId, onRemove) {
    var el = document.getElementById(containerId);
    var drop = dropId ? document.getElementById(dropId) : null;
    if (!el) return;
    el.innerHTML = "";
    if (drop) drop.classList.toggle("drop-zone--has-files", list.length > 0);
    for (var i = 0; i < list.length; i++) {
      var span = document.createElement("span");
      span.className = "file-chip";
      var f = list[i];
      if (f && f.type && f.type.indexOf("image/") === 0 && window.URL && URL.createObjectURL) {
        var thumb = document.createElement("img");
        thumb.className = "file-chip__thumb";
        thumb.alt = "";
        try {
          thumb.src = URL.createObjectURL(f);
          thumb.onload = function () { try { URL.revokeObjectURL(this.src); } catch (e) {} };
        } catch (e) {}
        span.appendChild(thumb);
      }
      var name = document.createElement("span");
      name.className = "file-chip__name";
      name.textContent = f.name;
      span.appendChild(name);
      if (onRemove) {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "file-chip__remove";
        btn.setAttribute("aria-label", "Убрать файл " + list[i].name);
        btn.textContent = "×";
        (function (idx) {
          btn.addEventListener("click", function () { onRemove(idx); });
        })(i);
        span.appendChild(btn);
      }
      el.appendChild(span);
    }
  }

  function clearFileInputs(inputs) {
    inputs.forEach(function (inp) {
      if (inp) inp.value = "";
    });
  }

  function wireUploadZone(opts) {
    var getList = opts.getList;
    var setList = opts.setList;
    var cameraInput = opts.cameraInput;
    var pickInput = opts.pickInput;
    var btnCamera = document.getElementById(opts.btnCameraId);
    var btnFile = document.getElementById(opts.btnFileId);
    var chipsId = opts.chipsId;
    var dropId = opts.dropId;
    var onChange = opts.onChange;

    function handlePick(input) {
      if (!input || !input.files || !input.files.length) return;
      var merged = mergeFilesIntoList(getList(), Array.prototype.slice.call(input.files), opts.maxFiles || 99);
      setList(merged);
      renderFileChips(getList(), chipsId, dropId, removeAt);
      clearFileInputs([cameraInput, pickInput]);
      if (onChange) onChange();
    }

    function removeAt(idx) {
      var cur = getList().slice();
      if (idx < 0 || idx >= cur.length) return;
      cur.splice(idx, 1);
      setList(cur);
      renderFileChips(getList(), chipsId, dropId, removeAt);
      if (onChange) onChange();
    }

    if (btnCamera && cameraInput) {
      btnCamera.addEventListener("click", function () { cameraInput.click(); });
      cameraInput.addEventListener("change", function () { handlePick(cameraInput); });
    }
    if (btnFile && pickInput) {
      btnFile.addEventListener("click", function () { pickInput.click(); });
      pickInput.addEventListener("change", function () { handlePick(pickInput); });
    }
  }

  function updateKzUploadStatus() {
    var el = document.getElementById("kz-upload-status");
    if (!el) return;
    if (!kzFilesList.length) {
      el.classList.add("hidden");
      el.textContent = "";
      return;
    }
    el.classList.remove("hidden");
    el.textContent = "Файл выбран: " + kzFilesList[0].name + " - готово к проверке.";
  }

  function updateBtn() {
    var ready = consentEl && consentEl.checked && kzFilesList.length > 0;
    if (btn) {
      btn.disabled = !ready;
      btn.textContent = ready && needsPaymentBeforeReview() ? "Оплатить и проверить" : "Проверить заключение";
    }
  }

  wireUploadZone({
    getList: function () { return kzFilesList; },
    setList: function (v) { kzFilesList = v; },
    cameraInput: kzCameraInput,
    pickInput: kzPickInput,
    btnCameraId: "kz-btn-camera",
    btnFileId: "kz-btn-file",
    chipsId: "kz-chips",
    dropId: "kz-drop",
    onChange: function () {
      updateKzUploadStatus();
      updateBtn();
    },
    maxFiles: 1,
  });
  wireUploadZone({
    getList: function () { return labFilesList; },
    setList: function (v) { labFilesList = v; },
    cameraInput: labCameraInput,
    pickInput: labPickInput,
    btnCameraId: "lab-btn-camera",
    btnFileId: "lab-btn-file",
    chipsId: "lab-chips",
    dropId: "lab-drop",
    maxFiles: 3,
  });
  if (consentEl) consentEl.addEventListener("change", updateBtn);

  if (localStorage.getItem(ONBOARD_KEY) === "1" && onboard) onboard.classList.add("hidden");
  var btnOnboard = document.getElementById("btn-onboard-ok");
  if (btnOnboard) btnOnboard.addEventListener("click", function () {
    localStorage.setItem(ONBOARD_KEY, "1");
    if (onboard) onboard.classList.add("hidden");
    if (formCard) {
      formCard.scrollIntoView({ behavior: "smooth", block: "start" });
      var firstUpload = document.getElementById("kz-btn-camera");
      if (firstUpload) firstUpload.focus();
    }
    if (statusEl) statusEl.textContent = "Выберите «Сделать фото» или «Загрузить файл» для заключения.";
  });

  function pillClass(s) {
    if (s === "ok") return "pill pill--ok";
    if (s === "concern") return "pill pill--concern";
    return "pill pill--attention";
  }
  function pillLabel(s) {
    if (s === "ok") return "В порядке";
    if (s === "concern") return "Обратите внимание";
    return "Стоит уточнить";
  }
  function trafficIcon(light) {
    if (light === "green") return "●";
    if (light === "red") return "●";
    return "●";
  }

  function loadChecklistState() {
    try { return JSON.parse(localStorage.getItem(CHECKLIST_KEY) || "{}"); } catch (e) { return {}; }
  }
  function saveChecklistItem(id, checked) {
    var st = loadChecklistState();
    st[id] = !!checked;
    localStorage.setItem(CHECKLIST_KEY, JSON.stringify(st));
    if (checked) track("checklist_item", { checked_count: Object.keys(st).filter(function (k) { return st[k]; }).length });
  }

  function saveReport(pr) {
    if (noHistoryMode) return;
    try { sessionStorage.setItem(REPORT_KEY, JSON.stringify(pr)); } catch (e) {}
  }
  function restoreReport() {
    try {
      var raw = sessionStorage.getItem(REPORT_KEY);
      if (!raw) return false;
      var pr = JSON.parse(raw);
      if (!pr || (!pr.plain_summary_ru && !pr.upload_mismatch)) return false;
      formCard.classList.add("hidden");
      resultCard.classList.remove("hidden");
      renderReport(pr);
      track("restore_report", { light: pr.traffic_light });
      return true;
    } catch (e) { return false; }
  }

  function scoreRingHtml(pct, light, compact) {
    var r = compact ? 40 : 52;
    var c = 2 * Math.PI * r;
    var off = c * (1 - (pct != null ? pct / 100 : 0));
    var color = light === "green" ? "#1a8a72" : light === "red" ? "#dc2626" : "#d97706";
    var cx = compact ? 44 : 60;
    return (
      '<circle cx="' + cx + '" cy="' + cx + '" r="' + r + '" fill="none" stroke="#e8f5f1" stroke-width="' + (compact ? 8 : 10) + '"/>' +
      '<circle cx="' + cx + '" cy="' + cx + '" r="' + r + '" fill="none" stroke="' + color + '" stroke-width="' + (compact ? 8 : 10) + '" ' +
      'stroke-dasharray="' + c + '" stroke-dashoffset="' + off + '" stroke-linecap="round" transform="rotate(-90 ' + cx + " " + cx + ')"/>' +
      '<text x="' + cx + '" y="' + (cx - 2) + '" text-anchor="middle" font-size="' + (compact ? "16" : "22") + '" font-weight="800" fill="#063d35">' +
      (pct != null ? pct + "%" : "-") + "</text>"
    );
  }

  function renderScoreRing(pct, light, label, compact) {
    var svg = document.getElementById("score-svg");
    var card = document.getElementById("score-card-wrap");
    if (!svg) return;
    if (card) card.classList.toggle("score-card--secondary", !!compact);
    var cx = compact ? 44 : 60;
    svg.setAttribute("viewBox", compact ? "0 0 88 88" : "0 0 120 120");
    svg.innerHTML = scoreRingHtml(pct, light, compact);
    var cap = document.getElementById("score-caption");
    if (cap) cap.textContent = label || "";
  }

  function renderTrafficPill(light, label) {
    var el = document.getElementById("traffic-pill");
    var wrap = document.getElementById("result-hero-wrap");
    if (!el) return;
    el.className = "traffic-pill traffic-pill--" + (light === "green" ? "green" : light === "red" ? "red" : "yellow");
    el.innerHTML = '<span aria-hidden="true">' + trafficIcon(light) + "</span> " + escapeHtml(label || "");
    el.classList.remove("hidden");
    if (wrap) wrap.classList.remove("hidden");
  }

  function renderQualityBanner(dq, light) {
    var el = document.getElementById("quality-banner");
    if (!el) return;
    if (!dq || !dq.hint_ru || dq.level === "good") { el.classList.add("hidden"); return; }
    el.className = "quality-banner quality-banner--" + (dq.level === "low" ? "low" : "medium");
    el.textContent = dq.hint_ru;
    el.classList.remove("hidden");
    if (dq.level === "low" && light === "green") el.textContent += " Не полагайтесь только на «зелёный» статус.";
  }

  function renderBlocksPanel(blocks) {
    var cards = document.getElementById("block-cards");
    var panel = document.getElementById("blocks-panel");
    if (!cards) return;
    cards.innerHTML = "";
    if (!blocks || !blocks.length) {
      if (panel) panel.classList.add("hidden");
      return;
    }
    if (panel) panel.classList.remove("hidden");
    blocks.forEach(function (b) {
      var article = document.createElement("article");
      article.className = "block-item block-item--" + (b.status || "attention");
      var scoreText = b.score_pct != null ? '<span class="block-item__pct">' + b.score_pct + "%</span>" : "";
      var gaps = b.gaps && b.gaps.length
        ? '<ul class="gap-list">' + b.gaps.map(function (g) { return "<li>" + escapeHtml(g) + "</li>"; }).join("") + "</ul>"
        : "";
      var protoLine = "";
      if (b.protocol_excerpt) {
        protoLine =
          '<div class="block-item__proto"><span class="block-item__proto-label">По протоколу Минздрава</span>' +
          '<p class="block-item__excerpt">' +
          escapeHtml(b.protocol_excerpt) +
          "</p></div>";
      }
      var comment = (b.summary_ru || b.why_ru || "-").trim();
      article.innerHTML =
        '<div class="block-item__cols">' +
        '<div class="block-item__name">' + escapeHtml(b.title) + "</div>" +
        '<div class="block-item__score">' + scoreText + '<span class="' + pillClass(b.status) + '">' + pillLabel(b.status) + "</span></div>" +
        "</div>" +
        '<div class="block-item__comment">' +
        '<span class="block-item__comment-label">Комментарий</span>' +
        '<p class="block-item__comment-text">' + escapeHtml(comment) + "</p>" +
        protoLine + gaps +
        "</div>";
      cards.appendChild(article);
    });
  }

  function loadQuestionTone() {
    try {
      var saved = localStorage.getItem(QUESTION_TONE_KEY);
      if (saved) selectedQuestionTone = normalizeQuestionToneId(saved);
    } catch (e) {}
  }

  function saveQuestionTone(tone) {
    selectedQuestionTone = normalizeQuestionToneId(tone || "serious");
    try { localStorage.setItem(QUESTION_TONE_KEY, selectedQuestionTone); } catch (e) {}
  }

  function renderTonePicker() {
    var wrap = document.getElementById("tone-chips");
    if (!wrap) return;
    wrap.innerHTML = "";
    questionTonesCatalog.forEach(function (t) {
      var btn = document.createElement("button");
      btn.type = "button";
      var active = selectedQuestionTone === t.id;
      btn.className = "tone-card tone-card--" + t.id + (active ? " tone-card--active" : "");
      btn.setAttribute("role", "radio");
      btn.setAttribute("aria-checked", active ? "true" : "false");
      btn.dataset.tone = t.id;
      if (t.accent) btn.style.setProperty("--tone-accent", t.accent);
      btn.title = t.description_ru || "";
      btn.innerHTML =
        '<span class="tone-card__glow" aria-hidden="true"></span>' +
        luxIconHtml(t.icon || t.emoji || t.id, "tone-card__icon lux-icon--tone") +
        '<span class="tone-card__copy">' +
        '<span class="tone-card__label">' + escapeHtml(t.label_ru || t.id) + "</span>" +
        '<span class="tone-card__desc">' + escapeHtml(t.description_ru || "") + "</span>" +
        "</span>" +
        '<span class="tone-card__check" aria-hidden="true">' + (active ? "✓" : "") + "</span>";
      btn.addEventListener("click", function () {
        saveQuestionTone(t.id);
        renderTonePicker();
        track("question_tone_pick", { tone: t.id });
      });
      wrap.appendChild(btn);
    });
  }

  function syncUploadFormatsFromApi() {
    fetch(window.location.origin + "/api/patient/status")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var fm = data && data.upload_formats;
        if (!fm) return;
        if (fm.accept) {
          if (kzPickInput) kzPickInput.setAttribute("accept", fm.accept);
          if (labPickInput) labPickInput.setAttribute("accept", fm.accept);
        }
        if (fm.hint_ru) {
          var kh = document.getElementById("kz-formats-hint");
          if (kh) kh.textContent = fm.hint_ru;
        }
        if (fm.max_files != null) {
          kzMaxFiles = Math.max(1, parseInt(fm.max_files, 10) || 1);
        }
      })
      .catch(function () {});
  }

  function syncQuestionTonesFromApi() {
    fetch(window.location.origin + "/api/patient/status")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data && data.question_tones && data.question_tones.length) {
          questionTonesCatalog = data.question_tones;
          if (data.default_question_tone) selectedQuestionTone = normalizeQuestionToneId(data.default_question_tone);
          loadQuestionTone();
          renderTonePicker();
        }
      })
      .catch(function () {});
  }

  function renderQuestionCards(items, pr) {
    var cl = document.getElementById("action-checklist");
    var section = document.getElementById("questions-section");
    if (!cl) return;
    cl.innerHTML = "";
    if (!items || !items.length) {
      if (section) section.classList.add("hidden");
      return;
    }
    if (section) {
      section.classList.remove("hidden");
      var tone = normalizeQuestionToneId((pr && pr.question_tone) || selectedQuestionTone || "serious");
      section.className = "report-panel report-panel--questions questions-panel questions-panel--tone-" + tone;
      var lead = document.getElementById("questions-panel-lead");
      if (lead) {
        lead.textContent =
          (pr && pr.questions_intro_ru) ||
          "Вопросы по вашему заключению - коротко и по делу. Отметьте обсуждённые на приёме.";
      }
      var etiquette = document.getElementById("questions-panel-etiquette");
      if (etiquette) {
        etiquette.textContent =
          (pr && pr.questions_etiquette_ru) ||
          "Нажмите галочку после разговора с врачом - список сохранится на устройстве.";
      }
      var emojiEl = document.getElementById("questions-panel-emoji");
      if (emojiEl) emojiEl.innerHTML = luxIconHtml("stethoscope", "lux-icon--hero");
      var badge = document.getElementById("questions-tone-badge");
      if (badge) badge.classList.add("hidden");
    }
    var checklistState = loadChecklistState();
    items.forEach(function (item, idx) {
      var li = document.createElement("li");
      var checked = !!checklistState[item.id];
      var sev = item.severity === "high" ? " question-card--high" : item.severity === "low" ? " question-card--low" : "";
      if (checked) li.className = "question-card checked" + sev;
      else li.className = "question-card" + sev;
      var cat = item.category_ru
        ? '<span class="question-card__cat">' + escapeHtml(item.category_ru) + "</span>"
        : "";
      var why = item.why_ru
        ? '<span class="question-card__why">' + escapeHtml(item.why_ru) + "</span>"
        : "";
      var ctx = item.plain_context
        ? '<span class="question-card__context">В заключении: ' + escapeHtml(item.plain_context) + "</span>"
        : "";
      li.innerHTML =
        '<label class="question-card__label" for="ck-' +
        escapeHtml(item.id) +
        '">' +
        '<div class="question-card__shell">' +
        '<span class="question-card__num" aria-hidden="true">' +
        (idx + 1) +
        "</span>" +
        '<span class="question-card__body">' +
        cat +
        '<span class="question-card__text">' +
        escapeHtml(item.text || item.title || "") +
        "</span>" +
        why +
        ctx +
        "</span>" +
        '<input type="checkbox" class="question-card__check" id="ck-' +
        escapeHtml(item.id) +
        '" ' +
        (checked ? "checked" : "") +
        ' aria-label="Обсудили с врачом" />' +
        "</div></label>";
      var cb = li.querySelector("input");
      cb.addEventListener("change", function () {
        saveChecklistItem(item.id, cb.checked);
        li.classList.toggle("checked", cb.checked);
        track("checklist_item", { checked: cb.checked, intent: item.intent || item.block_id });
      });
      cl.appendChild(li);
    });
  }

  function renderOncoQuestions(block) {
    var wrap = document.getElementById("onco-questions-wrap");
    if (!wrap) return;
    var list = document.getElementById("onco-questions-list");
    var qs = (block && block.questions) || [];
    if (!block || !qs.length) {
      wrap.classList.add("hidden");
      if (list) list.innerHTML = "";
      return;
    }
    var intro = document.getElementById("onco-questions-intro");
    if (intro) intro.textContent = block.intro_ru || "";
    var note = document.getElementById("onco-questions-note");
    if (note) note.textContent = block.disclaimer_ru || "";
    if (list) {
      list.innerHTML = "";
      qs.forEach(function (q) {
        var li = document.createElement("li");
        var ic = document.createElement("span");
        ic.className = "onco-ic";
        ic.setAttribute("aria-hidden", "true");
        ic.innerHTML = "&#10003;";
        var span = document.createElement("span");
        span.textContent = q;
        li.appendChild(ic);
        li.appendChild(span);
        list.appendChild(li);
      });
    }
    wrap.classList.remove("hidden");
  }

  function renderP2Narratives(pr) {
    var wrap = document.getElementById("p2-narratives");
    if (!wrap) return;
    wrap.innerHTML = "";
    var items = pr.plain_narratives || [];
    if (!items.length) { wrap.classList.add("hidden"); return; }
    wrap.classList.remove("hidden");
    var html = '<div class="section-head"><span class="section-dot"></span><h2>Пояснения простым языком</h2></div>';
    items.forEach(function (n) {
      html += '<div class="p2-narrative"><strong>' + escapeHtml(n.title || "") + "</strong>" + escapeHtml(n.text_ru || "") + "</div>";
    });
    wrap.innerHTML = html;
  }

  function renderLabPanel(lc) {
    var box = document.getElementById("lab-result");
    if (!box) return;
    if (!lc || !lc.lab_count) { box.classList.add("hidden"); box.innerHTML = ""; return; }
    box.classList.remove("hidden");
    var html = '<details class="cites-fold"><summary>Сверка с анализами</summary><div class="lab-panel" style="margin-top:0.45rem">';
    if (lc.summary_ru) html += '<p class="lab-summary">' + escapeHtml(lc.summary_ru) + "</p>";
    if (lc.panels_ru && lc.panels_ru.length) {
      html += '<p style="font-size:0.72rem;color:var(--muted);margin:0 0 0.45rem">Бланки: ' + escapeHtml(lc.panels_ru.join(", ")) + "</p>";
    }
    var rows = lc.markers_table || [];
    if (rows.length) {
      html += '<table class="lab-table"><thead><tr><th>Показатель</th><th>Результат</th><th>В заключении</th></tr></thead><tbody>';
      rows.forEach(function (r) {
        var val = r.value != null && r.value !== "" ? String(r.value) : "-";
        if (r.unit) val += " " + r.unit;
        if (r.flag === "high") val += " ↑";
        html += "<tr><td>" + escapeHtml(r.marker || "-") + "</td><td>" + escapeHtml(val) + "</td><td>" + (r.in_kz ? "да" : "нет") + "</td></tr>";
      });
      html += "</tbody></table>";
    }
    if (lc.missing_in_kz_lines && lc.missing_in_kz_lines.length) {
      html += '<p class="lab-miss-title">Не названы в тексте заключения</p><ul style="margin:0;padding-left:1.1rem;font-size:0.78rem">';
      lc.missing_in_kz_lines.forEach(function (line) { html += "<li>" + escapeHtml(line) + "</li>"; });
      html += "</ul>";
    }
    html += "</div></details>";
    box.innerHTML = html;
  }

  function renderProtocolPanel(pc) {
    var box = document.getElementById("protocol-result");
    if (!box) return;
    if (!pc || !(pc.missing_recommended_exams && pc.missing_recommended_exams.length)) {
      box.classList.add("hidden"); box.innerHTML = ""; return;
    }
    box.classList.remove("hidden");
    var html = '<section class="report-panel report-panel--protocol"><div class="section-head"><span class="section-dot"></span><h2>Требования протокола</h2></div>';
    if (pc.protocol_title || pc.protocol_link) {
      html += '<p class="protocol-panel-note">См. протокол Минздрава в блоке выше.</p>';
    }
    pc.missing_recommended_exams.forEach(function (m) {
      html += '<div class="protocol-req block-item block-item--concern"><strong>' + escapeHtml(m.exam_name || "Обследование") + "</strong>";
      html += "<p>" + escapeHtml(m.patient_note_ru || "") + "</p></div>";
    });
    html += "</section>";
    box.innerHTML = html;
  }

  function saveHistory(pr) {
    if (noHistoryMode) return;
    try {
      var list = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
      list.unshift({ ts: new Date().toISOString(), pct: pr.overall_pct, light: pr.traffic_light, label: pr.overall_label_ru, summary: pr.plain_summary_ru || "" });
      localStorage.setItem(HISTORY_KEY, JSON.stringify(list.slice(0, 5)));
      renderHistory();
      syncCloudHistory(list.slice(0, 5));
    } catch (e) {}
  }

  function syncCloudHistory(list) {
    var tok = localStorage.getItem(SESSION_KEY);
    if (!tok) return;
    fetch(window.location.origin + "/api/patient/account/sync", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_token: tok, history: list }),
    }).catch(function () {});
  }

  function renderHistory() {
    var wrap = document.getElementById("history-wrap");
    var listEl = document.getElementById("history-list");
    if (!wrap || !listEl) return;
    try {
      var list = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
      if (!list.length) { wrap.classList.add("hidden"); return; }
      wrap.classList.remove("hidden");
      listEl.innerHTML = list.map(function (it) {
        var d = new Date(it.ts);
        return '<div class="history-item">' + d.toLocaleString("ru-RU") + " · " + (it.pct != null ? it.pct + "%" : "-") + " · " + escapeHtml(it.label || "") + "</div>";
      }).join("");
    } catch (e) { wrap.classList.add("hidden"); }
  }
  renderHistory();

  function buildShareText(pr) {
    var lines = ["Проверь КЗ - лист на приём", ""];
    if (pr.headline_ru) lines.push(pr.headline_ru);
    if (pr.plain_summary_ru) lines.push(pr.plain_summary_ru);
    var qs = pr.questions_for_doctor || [];
    if (qs.length) {
      lines.push("", "Вопросы врачу:");
      qs.forEach(function (q, i) { lines.push((i + 1) + ". " + q); });
    }
    lines.push("", "Не является диагнозом.");
    return lines.join("\n");
  }

  function normalizePatientReport(raw) {
    var pr = raw || {};
    if (pr.top_summary && pr.scores) return pr;
    if (pr.report_schema_version >= 2 && pr.top_summary) return pr;
    var legacy = {
      report_schema_version: pr.report_schema_version || 1,
      headline_ru: pr.headline_ru || pr.overall_label_ru || "",
      plain_summary_ru: pr.plain_summary_ru || "",
      top_summary: {
        headline_ru: pr.headline_ru || pr.overall_label_ru || "",
        plain_summary_ru: pr.plain_summary_ru || "",
        main_takeaway_ru: "",
      },
      scores: {
        document_completeness: { pct: pr.overall_pct, label_ru: "Полнота КЗ", hint_ru: pr.overall_label_ru || "" },
        patient_clarity: { pct: pr.overall_pct, label_ru: "Понятность", hint_ru: "" },
        protocol_match_confidence: { pct: null, label_ru: "Уверенность КП", hint_ru: "" },
      },
      show_single_overall_score: true,
      understood_from_document: (pr.document_read_back_ru || []).map(function (line) {
        return { label_ru: "Факт", value_ru: line };
      }),
      clarification_points: (pr.priority_topics || []).map(function (t) {
        return { topic_ru: t.topic, text_ru: t.why_ru };
      }),
      message_to_doctor: { text_ru: (pr.questions_for_doctor || []).slice(0, 3).join(" ") },
      visit_sheet: { text_ru: buildShareText(pr) },
      plain_terms: [],
    };
    return Object.assign({}, pr, legacy);
  }

  function renderTopSummary(report) {
    var top = report.top_summary || {};
    var hl = document.getElementById("headline-ru");
    var ps = document.getElementById("plain-summary");
    var mt = document.getElementById("main-takeaway");
    var pcn = document.getElementById("protocol-confidence-note");
    if (hl) hl.textContent = top.headline_ru || report.headline_ru || "";
    if (ps) ps.textContent = top.plain_summary_ru || report.plain_summary_ru || "";
    if (mt) {
      if (top.main_takeaway_ru) {
        mt.textContent = top.main_takeaway_ru;
        mt.classList.remove("hidden");
      } else mt.classList.add("hidden");
    }
    if (pcn) {
      var note = top.protocol_confidence_note_ru || "";
      if (note) {
        pcn.textContent = note;
        pcn.classList.remove("hidden");
        track("low_confidence_shown", { bucket: report.protocol_confidence_bucket });
      } else pcn.classList.add("hidden");
    }
  }

  function renderScoreCards(report) {
    var wrap = document.getElementById("score-cards-wrap");
    var hero = document.getElementById("score-card-wrap");
    if (!wrap) return;
    wrap.innerHTML = "";
    var scores = report.scores || {};
    var keys = ["document_completeness", "patient_clarity", "protocol_match_confidence"];
    var any = false;
    keys.forEach(function (k) {
      var sc = scores[k];
      if (!sc) return;
      any = true;
      var div = document.createElement("div");
      div.className = "score-card-mini";
      div.innerHTML =
        '<div class="score-card-mini__label">' + escapeHtml(sc.label_ru || k) + "</div>" +
        '<div class="score-card-mini__pct">' + (sc.pct != null ? sc.pct + "%" : "-") + "</div>" +
        '<p class="score-card-mini__hint">' + escapeHtml(sc.hint_ru || "") + "</p>";
      wrap.appendChild(div);
    });
    if (hero) {
      if (report.show_single_overall_score === false || any) {
        hero.classList.add("hidden");
      } else {
        hero.classList.remove("hidden");
        renderScoreRing(report.overall_pct, report.traffic_light, "Сводная оценка", true);
      }
    }
  }

  function renderUnderstoodFromDocument(report) {
    var ul = document.getElementById("understood-list");
    var wrap = document.getElementById("understood-wrap");
    if (!ul || !wrap) return;
    ul.innerHTML = "";
    var items = report.understood_from_document || [];
    if (!items.length) {
      wrap.classList.add("hidden");
      return;
    }
    wrap.classList.remove("hidden");
    items.forEach(function (it) {
      var li = document.createElement("li");
      li.textContent = (it.label_ru ? it.label_ru + ": " : "") + (it.value_ru || "");
      ul.appendChild(li);
    });
  }

  function renderClarificationPoints(report) {
    var ul = document.getElementById("clarify-list");
    var wrap = document.getElementById("clarify-wrap");
    if (!ul || !wrap) return;
    ul.innerHTML = "";
    var items = report.clarification_points || [];
    if (!items.length) {
      wrap.classList.add("hidden");
      return;
    }
    wrap.classList.remove("hidden");
    items.forEach(function (it) {
      var li = document.createElement("li");
      li.textContent = it.text_ru || it.topic_ru || "";
      ul.appendChild(li);
    });
  }

  function renderMessageToDoctor(report) {
    var listEl = document.getElementById("message-doctor-list");
    var el = document.getElementById("message-doctor-text");
    var wrap = document.getElementById("message-doctor-wrap");
    var msg = report.message_to_doctor || {};
    var items = report.action_checklist || report.questions_structured || [];
    var lines = [];
    if (items.length) {
      items.forEach(function (it) {
        var t = (it && (it.text || it.title)) || "";
        if (t) lines.push(t);
      });
    } else if (msg.text_ru) {
      lines = String(msg.text_ru).split(/\?\s+/).filter(Boolean).map(function (s) {
        return s.trim().endsWith("?") ? s.trim() : s.trim() + "?";
      });
    }
    if (!listEl || !wrap) return;
    if (!lines.length) {
      wrap.classList.add("hidden");
      if (listEl) listEl.innerHTML = "";
      if (el) el.textContent = "";
      return;
    }
    wrap.classList.remove("hidden");
    listEl.innerHTML = "";
    lines.slice(0, 5).forEach(function (line) {
      var li = document.createElement("li");
      li.textContent = line;
      listEl.appendChild(li);
    });
    if (el) el.textContent = lines.join("\n");
  }

  function renderVisitSheet(report) {
    var el = document.getElementById("visit-sheet-text");
    var wrap = document.getElementById("visit-sheet-wrap");
    var vs = report.visit_sheet || {};
    if (!el || !wrap) return;
    if (!vs.text_ru) {
      wrap.classList.add("hidden");
      return;
    }
    wrap.classList.remove("hidden");
    el.textContent = vs.text_ru;
  }

  function renderPlainTerms(report) {
    var box = document.getElementById("plain-terms-list");
    var wrap = document.getElementById("plain-terms-wrap");
    if (!box || !wrap) return;
    box.innerHTML = "";
    var terms = report.plain_terms || [];
    if (!terms.length) {
      wrap.classList.add("hidden");
      return;
    }
    wrap.classList.remove("hidden");
    terms.forEach(function (t, idx) {
      var chip = document.createElement("button");
      chip.type = "button";
      chip.className = "term-chip";
      chip.setAttribute("aria-expanded", "false");
      chip.id = "term-chip-" + idx;
      chip.textContent = t.term || "";
      var expl = document.createElement("div");
      expl.className = "term-explanation hidden";
      expl.textContent = t.explanation_ru || "";
      chip.addEventListener("click", function () {
        var open = chip.getAttribute("aria-expanded") === "true";
        chip.setAttribute("aria-expanded", open ? "false" : "true");
        expl.classList.toggle("hidden", open);
        if (!open) track("term_expanded", { term: t.term });
      });
      box.appendChild(chip);
      box.appendChild(expl);
    });
  }

  function splitSentences(text) {
    var str = String(text || "");
    var out = [];
    var buf = "";
    for (var i = 0; i < str.length; i++) {
      buf += str[i];
      if (/[.!?]/.test(str[i]) && (i + 1 >= str.length || str[i + 1] === " ")) {
        out.push(buf.trim());
        buf = "";
      }
    }
    if (buf.trim()) out.push(buf.trim());
    return out.filter(Boolean);
  }

  function renderRedFlags(report) {
    var wrap = document.getElementById("red-flags-wrap");
    var body = document.getElementById("red-flags-body");
    if (!wrap || !body) return;
    var text = report.red_flags_ru || "";
    if (!text) { wrap.classList.add("hidden"); return; }
    var sentences = splitSentences(text);
    var disclaimer = "";
    if (sentences.length > 1 && /справочн|не диагноз/i.test(sentences[sentences.length - 1])) {
      disclaimer = sentences.pop();
    }
    var html = "";
    if (sentences.length) {
      html += "<ul>" + sentences.map(function (s) { return "<li>" + escapeHtml(s) + "</li>"; }).join("") + "</ul>";
    } else {
      html += "<p>" + escapeHtml(text) + "</p>";
    }
    if (disclaimer) {
      html += '<p class="red-flags-disclaimer" style="font-size:0.74rem;color:var(--muted);margin:0.45rem 0 0">' + escapeHtml(disclaimer) + "</p>";
    }
    body.innerHTML = html;
    wrap.classList.remove("hidden");
  }

  function renderProtocolSummaryPanel(report) {
    var wrap = document.getElementById("protocol-summary-wrap");
    var intro = document.getElementById("protocol-summary-intro");
    var list = document.getElementById("protocol-summary-list");
    if (!wrap || !list) return;
    var panel = report.protocol_summary_panel;
    if (!panel || !panel.items || !panel.items.length) {
      wrap.classList.add("hidden");
      return;
    }
    var title = document.getElementById("protocol-summary-title");
    if (title && panel.title_ru) title.textContent = panel.title_ru;
    if (intro) intro.textContent = panel.intro_ru || "";
    list.innerHTML = "";
    panel.items.forEach(function (it) {
      var li = document.createElement("li");
      var yes = !!it.present;
      li.innerHTML =
        '<span class="protocol-summary__mark ' + (yes ? "protocol-summary__mark--yes" : "protocol-summary__mark--no") + '" aria-hidden="true">' +
        (yes ? "✓" : "•") + "</span>" +
        '<span><strong>' + escapeHtml(it.name_ru || "") + "</strong>" +
        (it.note_ru ? ' <span class="protocol-summary__note">- ' + escapeHtml(it.note_ru) + "</span>" : "") +
        "</span>";
      list.appendChild(li);
    });
    wrap.classList.remove("hidden");
  }

  function renderQuestionsMore(report) {
    var el = document.getElementById("questions-more");
    if (!el) return;
    var hidden = report.questions_hidden_count || 0;
    if (report.questions_truncated && hidden > 0) {
      el.textContent = "Ещё " + hidden + " вопрос(ов) - в полном разборе. Сейчас показаны главные.";
      el.classList.remove("hidden");
    } else {
      el.classList.add("hidden");
      el.textContent = "";
    }
  }

  function renderClinicTrust() {
    var el = document.getElementById("clinic-trust");
    if (!el) return;
    if (clinicConfig && clinicConfig.footer_ru) {
      el.textContent = clinicConfig.footer_ru;
      el.classList.remove("hidden");
    } else {
      el.classList.add("hidden");
    }
  }

  function copyText(text, eventName) {
    if (!text || !navigator.clipboard) return;
    navigator.clipboard.writeText(text).then(function () {
      if (statusEl) statusEl.textContent = "Скопировано.";
      track(eventName || "question_copied");
    });
  }

  function renderUploadJokeCard(pr) {
    resetResultViewForMismatch();
    var el = document.getElementById("upload-joke-card");
    var body = document.getElementById("result-body");
    if (!el) return;
    var joke = pr.upload_joke || {};
    var emoji = joke.emoji || "🤔";
    var title = joke.title_ru || pr.headline_ru || "Это не тот документ";
    var text = joke.body_ru || pr.plain_summary_ru || "";
    var guess = joke.guessed_what_ru ? '<span class="upload-joke__guess">Похоже на: ' + escapeHtml(joke.guessed_what_ru) + "</span>" : "";
    var hint = joke.hint_ru || "";
    el.classList.remove("hidden");
    el.innerHTML =
      '<div class="upload-joke__inner">' +
      '<div class="upload-joke__emoji" aria-hidden="true">' + emoji + "</div>" +
      '<h2 class="upload-joke__title">' + escapeHtml(title) + "</h2>" +
      guess +
      '<p class="upload-joke__body">' + escapeHtml(text) + "</p>" +
      (hint ? '<p class="upload-joke__hint">' + escapeHtml(hint) + "</p>" : "") +
      "</div>";
    if (body) body.classList.add("hidden");
    setAgainButtonVisible(true, "Загрузить правильный документ");
  }

  function resetResultViewForMismatch() {
    var qb = document.getElementById("quality-banner");
    if (qb) {
      qb.classList.add("hidden");
      qb.textContent = "";
    }
    var body = document.getElementById("result-body");
    if (body) body.classList.add("hidden");
    lastProtocolLinks = [];
    renderProtocolStrip([]);
    ["headline-ru", "plain-summary", "main-takeaway", "protocol-confidence-note"].forEach(function (id) {
      var el = document.getElementById(id);
      if (!el) return;
      el.textContent = "";
      if (id !== "headline-ru" && id !== "plain-summary") el.classList.add("hidden");
    });
    var mb = document.getElementById("matched-badge");
    if (mb) {
      mb.classList.add("hidden");
      mb.innerHTML = "";
      mb.setAttribute("aria-hidden", "true");
    }
    var tp = document.getElementById("traffic-pill");
    if (tp) {
      tp.className = "traffic-pill traffic-pill--yellow hidden";
      tp.textContent = "";
    }
    var scoreWrap = document.getElementById("score-cards-wrap");
    if (scoreWrap) scoreWrap.innerHTML = "";
    var hero = document.getElementById("score-card-wrap");
    if (hero) hero.classList.add("hidden");
    renderBlocksPanel([]);
    renderProtocolPanel(null);
    var labBox = document.getElementById("lab-result");
    if (labBox) {
      labBox.classList.add("hidden");
      labBox.innerHTML = "";
    }
    var cites = document.getElementById("cites-wrap");
    if (cites) cites.innerHTML = "";
    var citesDetails = document.getElementById("cites-details");
    if (citesDetails) citesDetails.hidden = true;
    var jokeCard = document.getElementById("upload-joke-card");
    if (jokeCard) jokeCard.classList.remove("hidden");
  }

  function renderReport(pr) {
    pr = normalizePatientReport(pr);
    lastReport = pr;
    lastProtocolLinks = pr.protocol_links || (pr.primary_protocol ? [pr.primary_protocol] : []);
    reviewFingerprint = pr.review_fingerprint || null;
    saveReport(pr);

    var jokeCard = document.getElementById("upload-joke-card");
    var resultBody = document.getElementById("result-body");
    if (pr.upload_mismatch) {
      renderUploadJokeCard(pr);
      track("report_view", { upload_mismatch: true, kind: pr.guessed_kind || pr.mismatch_slot });
      return;
    }
    if (jokeCard) {
      jokeCard.classList.add("hidden");
      jokeCard.innerHTML = "";
    }
    if (resultBody) resultBody.classList.remove("hidden");
    var hl = document.getElementById("headline-ru");
    if (hl && !pr.top_summary) hl.textContent = pr.headline_ru || pr.overall_label_ru || "";

    renderTopSummary(pr);
    renderTrafficPill(pr.traffic_light, pr.overall_label_ru);
    renderQualityBanner(pr.document_quality, pr.traffic_light);
    renderScoreCards(pr);
    renderProtocolStrip(lastProtocolLinks);

    var mb = document.getElementById("matched-badge");
    if (mb) mb.classList.add("hidden");

    var ps = document.getElementById("plain-summary");
    if (ps && !pr.top_summary) ps.textContent = pr.plain_summary_ru || "";

    renderRedFlags(pr);
    renderProtocolSummaryPanel(pr);
    renderClinicTrust();
    renderUnderstoodFromDocument(pr);
    renderClarificationPoints(pr);
    renderMessageToDoctor(pr);
    renderVisitSheet(pr);
    renderPlainTerms(pr);

    var ns = document.getElementById("next-steps");
    if (ns) {
      ns.innerHTML = "";
      var steps = pr.next_steps || pr.next_steps_ru || [];
      steps.forEach(function (s) {
        var li = document.createElement("li");
        li.textContent = typeof s === "string" ? s : s.step_ru || "";
        ns.appendChild(li);
      });
    }

    var rbw = document.getElementById("read-back-wrap");
    var rbl = document.getElementById("read-back-list");
    if (rbl) rbl.innerHTML = "";
    var rb = pr.document_read_back_ru || [];
    if (rbw && rbl && rb.length) {
      rbw.classList.remove("hidden");
      rb.forEach(function (line) {
        var li = document.createElement("li");
        li.textContent = line;
        rbl.appendChild(li);
      });
    } else if (rbw) rbw.classList.add("hidden");

    renderQuestionCards(pr.action_checklist || [], pr);
    renderQuestionsMore(pr);
    renderOncoQuestions(pr.onco_questions);

    var pw = document.getElementById("priority-wrap");
    var pl = document.getElementById("priority-list");
    if (pl) pl.innerHTML = "";
    var topics = pr.priority_topics || [];
    if (pw && pl) {
      if (topics.length) {
        pw.classList.remove("hidden");
        topics.forEach(function (t) {
          var li = document.createElement("li");
          if (t.severity === "high") li.className = "sev-high";
          li.innerHTML = "<strong>" + escapeHtml(t.topic) + "</strong><br />" + escapeHtml(t.why_ru || "");
          pl.appendChild(li);
        });
      } else pw.classList.add("hidden");
    }

    renderBlocksPanel(pr.blocks || []);

    renderP2Narratives(pr);
    renderLabPanel(pr.lab_crosscheck);
    renderProtocolPanel(pr.protocol_context);

    var cites = document.getElementById("cites-wrap");
    var citesDetails = document.getElementById("cites-details");
    if (cites) {
      cites.innerHTML = "";
      if (pr.protocol_citations && pr.protocol_citations.length && citesDetails) {
        citesDetails.hidden = false;
        pr.protocol_citations.forEach(function (c) {
          var div = document.createElement("div");
          div.className = "cite";
          var head = escapeHtml(c.protocol_title || "По протоколу Минздрава");
          if (c.section) head += ' <span class="cite__section">· ' + escapeHtml(c.section) + "</span>";
          div.innerHTML =
            '<div class="cite__head">' +
            head +
            '</div><p class="cite__text">' +
            escapeHtml(c.excerpt || "") +
            "</p>";
          cites.appendChild(div);
        });
      } else if (citesDetails) citesDetails.hidden = true;
    }

    var disc = document.getElementById("disclaimer");
    if (disc) {
      var t = pr.disclaimer_ru || "";
      if (pr.limitations_ru) t += " " + pr.limitations_ru;
      if (clinicConfig && clinicConfig.footer_ru) t += " " + clinicConfig.footer_ru;
      disc.textContent = t;
    }
    saveHistory(pr);
    track("report_view", {
      light: pr.traffic_light,
      pct: pr.overall_pct,
      block_count: (pr.blocks || []).length,
      protocol_confidence_bucket: pr.protocol_confidence_bucket,
    });
    track("patient_result_aha", {
      has_questions: !!(pr.questions_for_doctor && pr.questions_for_doctor.length),
      has_visit_sheet: !!(pr.visit_sheet && pr.visit_sheet.text_ru),
      protocol_confidence_bucket: pr.protocol_confidence_bucket,
    });
    setAgainButtonVisible(true);
  }

  function showLoader(stage) {
    loader.classList.remove("hidden");
    if (loaderText) loaderText.textContent = stage || "Анализируем документ";
  }
  function hideLoader() { loader.classList.add("hidden"); }

  function buildFormData() {
    var fd = new FormData();
    var i;
    for (i = 0; i < kzFilesList.length; i++) fd.append("files", kzFilesList[i]);
    for (i = 0; i < labFilesList.length; i++) fd.append("lab_files", labFilesList[i]);
    fd.append("consent", "1");
    var age = document.getElementById("age-years");
    var sex = document.getElementById("sex");
    if (age && age.value) fd.append("age_years", age.value);
    if (sex && sex.value) fd.append("sex", sex.value);
    if (clinicId) fd.append("clinic_id", clinicId);
    if (selectedTier) fd.append("tier_id", selectedTier);
    if (paidToken) fd.append("payment_token", paidToken);
    fd.append("question_tone", selectedQuestionTone || "serious");
    return fd;
  }

  function handleReviewResult(data) {
    hideLoader();
    if (btn) btn.disabled = false;
    formCard.classList.add("hidden");
    resultCard.classList.remove("hidden");
    var pr = (data && data.patient_report) || {};
    if (data && data.upload_mismatch) {
      pr.upload_mismatch = true;
      if (data.guessed_kind) pr.guessed_kind = data.guessed_kind;
      if (data.mismatch_slot) pr.mismatch_slot = data.mismatch_slot;
    }
    if (data && data.review_fingerprint) {
      pr.review_fingerprint = data.review_fingerprint;
      reviewFingerprint = data.review_fingerprint;
    }
    renderReport(pr);
    window.scrollTo({ top: 0, behavior: "smooth" });
    track("upload_done", {
      light: pr.traffic_light,
      latency_ms: data.latency_ms,
      upload_mismatch: !!(data && data.upload_mismatch) || !!pr.upload_mismatch,
      guessed_kind: data.guessed_kind || pr.guessed_kind,
    });
  }

  function runReviewFetch() {
    track("upload_start", { tier: selectedTier, lab_count: labFilesList.length });
    try {
      sessionStorage.removeItem(REPORT_KEY);
      reviewFingerprint = null;
    } catch (e) {}
    btn.disabled = true;
    showLoader("Анализируем документ");
    fetch(window.location.origin + "/api/patient/review", { method: "POST", body: buildFormData() })
      .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.detail || "Ошибка"); return j; }); })
      .then(handleReviewResult)
      .catch(function (err) {
        hideLoader();
        statusEl.textContent = err.message || "Не удалось выполнить проверку.";
        updateBtn();
      });
  }

  function runReviewSse() {
    track("upload_start", { tier: selectedTier, sse: true });
    try {
      sessionStorage.removeItem(REPORT_KEY);
      reviewFingerprint = null;
    } catch (e) {}
    btn.disabled = true;
    showLoader("Загрузка файлов…");
    fetch(window.location.origin + "/api/patient/review/stream", { method: "POST", body: buildFormData() })
      .then(function (resp) {
        if (!resp.ok || !resp.body) { runReviewFetch(); return; }
        var reader = resp.body.getReader();
        var dec = new TextDecoder();
        var buf = "";
        function pump() {
          return reader.read().then(function (chunk) {
            if (chunk.done) return;
            buf += dec.decode(chunk.value, { stream: true });
            var parts = buf.split("\n\n");
            buf = parts.pop() || "";
            parts.forEach(function (block) {
              var line = block.split("\n").find(function (l) { return l.indexOf("data:") === 0; });
              if (!line) return;
              try {
                var payload = JSON.parse(line.slice(5).trim());
                if (payload.type === "progress") {
                  showLoader(payload.label_ru);
                  if (loaderSub) loaderSub.textContent = (payload.pct != null ? payload.pct + "%" : "");
                }
                if (payload.type === "done" && payload.result) handleReviewResult(payload.result);
                if (payload.type === "error") throw new Error(payload.detail || "Ошибка");
              } catch (e) {
                if (e.message && e.message.indexOf("JSON") < 0) {
                  hideLoader();
                  statusEl.textContent = e.message;
                  updateBtn();
                }
              }
            });
            return pump();
          });
        }
        return pump();
      })
      .catch(function () { runReviewFetch(); });
  }

  if (btn) btn.addEventListener("click", function () {
    if (!kzFilesList.length) return;
    if (needsPaymentBeforeReview()) {
      btn.disabled = true;
      startPaymentSession(false);
      return;
    }
    if (useSse) runReviewSse(); else runReviewFetch();
  });

  function buildPrintSheet() {
    var host = document.getElementById("print-sheet");
    if (!host) return;
    var pr = lastReport || {};
    var parts = ["<h1>Лист на приём</h1>"];
    var meta = [new Date().toLocaleDateString("ru-RU")];
    if (clinicConfig && clinicConfig.name_ru) meta.push(clinicConfig.name_ru);
    parts.push('<p class="print-meta">' + escapeHtml(meta.join(" · ")) + "</p>");

    var top = pr.top_summary || {};
    var summary = top.plain_summary_ru || pr.plain_summary_ru || "";
    if (summary) parts.push("<h2>Краткий контекст</h2><p>" + escapeHtml(summary) + "</p>");

    var clarify = pr.clarification_points || [];
    if (clarify.length) {
      parts.push("<h2>Что уточнить</h2><ul>" + clarify.map(function (c) {
        return "<li>" + escapeHtml(c.text_ru || c.topic_ru || "") + "</li>";
      }).join("") + "</ul>");
    }

    var qs = (pr.action_checklist && pr.action_checklist.length)
      ? pr.action_checklist.map(function (q) { return q.text || q.title || ""; })
      : (pr.questions_for_doctor || []);
    qs = qs.filter(Boolean);
    if (qs.length) {
      parts.push("<h2>Вопросы врачу</h2><ol>" + qs.map(function (q) {
        return "<li>" + escapeHtml(q) + "</li>";
      }).join("") + "</ol>");
    }

    if (pr.red_flags_ru) {
      parts.push("<h2>Когда обратиться срочно</h2><p>" + escapeHtml(pr.red_flags_ru) + "</p>");
    }

    parts.push(
      "<h2>Что взять с собой</h2><ul>" +
      "<li>Заключение (КЗ)</li>" +
      "<li>Результаты обследований и анализов, если уже выполнены</li>" +
      "<li>Список принимаемых препаратов</li></ul>"
    );
    parts.push('<p class="print-disclaimer">' + escapeHtml(
      pr.disclaimer_ru ||
      "Protocol помогает понять документ и подготовить вопросы. Не является диагнозом и не заменяет врача."
    ) + "</p>");
    host.innerHTML = parts.join("");
  }

  var btnPrint = document.getElementById("btn-print");
  if (btnPrint) btnPrint.addEventListener("click", function () { track("print_tap"); buildPrintSheet(); window.print(); });
  var btnPrintVisit = document.getElementById("btn-print-visit");
  if (btnPrintVisit) btnPrintVisit.addEventListener("click", function () {
    track("visit_sheet_downloaded");
    buildPrintSheet();
    window.print();
  });

  var speaking = false;
  function readAloudText() {
    var pr = lastReport || {};
    var top = pr.top_summary || {};
    var bits = [];
    if (top.headline_ru || pr.headline_ru) bits.push(top.headline_ru || pr.headline_ru);
    if (top.plain_summary_ru || pr.plain_summary_ru) bits.push(top.plain_summary_ru || pr.plain_summary_ru);
    var qs = (pr.action_checklist || []).map(function (q) { return q.text || q.title || ""; }).filter(Boolean);
    if (qs.length) {
      bits.push("Вопросы врачу.");
      qs.forEach(function (q, i) { bits.push((i + 1) + ". " + q); });
    }
    return bits.join(" ");
  }
  var btnRead = document.getElementById("btn-read-aloud");
  function stopReading() {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    speaking = false;
    if (btnRead) { btnRead.setAttribute("aria-pressed", "false"); btnRead.textContent = "Прочитать вслух"; }
  }
  if (btnRead) {
    if (!("speechSynthesis" in window) || typeof SpeechSynthesisUtterance === "undefined") {
      btnRead.classList.add("hidden");
    } else {
      btnRead.addEventListener("click", function () {
        if (speaking) { stopReading(); return; }
        var text = readAloudText();
        if (!text) return;
        var u = new SpeechSynthesisUtterance(text);
        u.lang = "ru-RU";
        u.rate = 0.95;
        u.onend = function () { stopReading(); };
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
        speaking = true;
        btnRead.setAttribute("aria-pressed", "true");
        btnRead.textContent = "Остановить";
        track("read_aloud");
      });
    }
  }

  var FONT_KEY = "protocol_patient_font_larger";
  function applyFontPref() {
    var on = localStorage.getItem(FONT_KEY) === "1";
    document.body.classList.toggle("font-larger", on);
    var b = document.getElementById("btn-font-larger");
    if (b) b.setAttribute("aria-pressed", on ? "true" : "false");
  }
  var btnFont = document.getElementById("btn-font-larger");
  if (btnFont) btnFont.addEventListener("click", function () {
    var on = localStorage.getItem(FONT_KEY) === "1";
    localStorage.setItem(FONT_KEY, on ? "0" : "1");
    applyFontPref();
    track("font_larger_toggle", { on: !on });
  });
  applyFontPref();

  function checkReminderDue() {
    try {
      var raw = localStorage.getItem(REMINDER_KEY);
      if (!raw) return;
      var when = parseInt(raw, 10);
      if (when && Date.now() >= when) {
        if (statusEl) statusEl.textContent = "Напоминание: не забудьте обсудить вопросы с врачом на приёме.";
        localStorage.removeItem(REMINDER_KEY);
        track("reminder_due");
      }
    } catch (e) {}
  }
  checkReminderDue();

  var btnCopyMsg = document.getElementById("btn-copy-message");
  if (btnCopyMsg) btnCopyMsg.addEventListener("click", function () {
    if (!lastReport || !lastReport.message_to_doctor) return;
    copyText(lastReport.message_to_doctor.text_ru, "message_copied");
  });
  var btnShareMsg = document.getElementById("btn-share-message");
  if (btnShareMsg) btnShareMsg.addEventListener("click", function () {
    if (!lastReport || !lastReport.message_to_doctor) return;
    var text = lastReport.message_to_doctor.text_ru;
    if (navigator.share) navigator.share({ title: "Сообщение врачу", text: text }).catch(function () {});
    else copyText(text, "share_clicked");
  });
  var btnCopyVisit = document.getElementById("btn-copy-visit");
  if (btnCopyVisit) btnCopyVisit.addEventListener("click", function () {
    if (!lastReport || !lastReport.visit_sheet) return;
    copyText(lastReport.visit_sheet.text_ru, "visit_sheet_copied");
  });

  var btnShare = document.getElementById("btn-share");
  if (btnShare) btnShare.addEventListener("click", function () {
    if (!lastReport) return;
    track("share_tap");
    var text = buildShareText(lastReport);
    if (navigator.share) navigator.share({ title: "Проверь КЗ", text: text }).catch(function () {});
    else if (navigator.clipboard) navigator.clipboard.writeText(text).then(function () { statusEl.textContent = "Скопировано."; });
  });

  var btnAgain = document.getElementById("btn-again");
  if (btnAgain) btnAgain.addEventListener("click", function () {
    stopReading();
    setAgainButtonVisible(false);
    resultCard.classList.add("hidden");
    formCard.classList.remove("hidden");
    sessionStorage.removeItem(REPORT_KEY);
    kzFilesList = [];
    labFilesList = [];
    clearFileInputs([kzCameraInput, kzPickInput, labCameraInput, labPickInput]);
    renderFileChips(kzFilesList, "kz-chips", "kz-drop");
    renderFileChips(labFilesList, "lab-chips", "lab-drop");
    updateBtn();
  });

  var btnClear = document.getElementById("btn-clear-data");
  if (btnClear) btnClear.addEventListener("click", function () {
    localStorage.removeItem(HISTORY_KEY);
    localStorage.removeItem(CHECKLIST_KEY);
    sessionStorage.removeItem(REPORT_KEY);
    renderHistory();
    statusEl.textContent = "Данные на устройстве удалены.";
  });

  var btnReminder = document.getElementById("btn-set-reminder");
  if (btnReminder) btnReminder.addEventListener("click", function () {
    var when = Date.now() + 48 * 3600 * 1000;
    localStorage.setItem(REMINDER_KEY, String(when));
    track("reminder_set");
    statusEl.textContent = "Напоминание сохранено на этом устройстве (48 ч).";
  });

  function loadClinic() {
    var url = window.location.origin + "/api/patient/clinic" + (clinicId ? "?clinic_id=" + encodeURIComponent(clinicId) : "");
    fetch(url).then(function (r) { return r.json(); }).then(function (data) {
      if (!data || !data.clinic) return;
      clinicConfig = data.clinic;
      var banner = document.getElementById("clinic-banner");
      if (banner) {
        banner.textContent = clinicConfig.name_ru + " · " + (clinicConfig.tagline_ru || "");
        banner.classList.remove("hidden");
      }
      if (clinicConfig.primary_color) document.documentElement.style.setProperty("--g600", clinicConfig.primary_color);
      var brand = document.querySelector(".brand-lockup__name");
      if (brand && clinicId) brand.textContent = clinicConfig.name_ru;
      var brandTag = document.querySelector(".brand-lockup__tag");
      if (brandTag && clinicId && clinicConfig.tagline_ru) brandTag.textContent = clinicConfig.tagline_ru;
      if (!tierId && clinicConfig.default_tier) selectedTier = clinicConfig.default_tier;
      syncPatientMonetizationFromApi();
    }).catch(function () {});
  }

  function applyMonetizationState(mon) {
    if (!mon) return;
    monetization = mon;
    if (mon.default_tier_id && !tierId) selectedTier = mon.default_tier_id;
    if (mon.payment_required && paidToken) {
      var payNote = document.getElementById("payment-note");
      if (payNote) payNote.classList.add("payment-note--paid");
    }
    renderMonetizationUi();
  }

  function renderMonetizationUi() {
    var wrap = document.getElementById("tier-wrap");
    var banner = document.getElementById("patient-value-banner");
    var payNote = document.getElementById("payment-note");
    if (banner) {
      if (monetization.value_banner_ru) {
        banner.textContent = monetization.value_banner_ru;
        banner.classList.remove("hidden");
      } else {
        banner.classList.add("hidden");
        banner.textContent = "";
      }
    }
    if (wrap) {
      if (monetization.monetization_enabled && monetization.show_tier_picker) {
        wrap.classList.remove("hidden");
      } else {
        wrap.classList.add("hidden");
      }
    }
    if (payNote) {
      payNote.textContent = monetization.payment_note_ru || "";
      payNote.classList.toggle("hidden", !payNote.textContent);
    }
    renderTierBar(monetization.tiers || []);
    updateBtn();
  }

  function renderTierBar(tiers) {
    var bar = document.getElementById("tier-bar");
    if (!bar) return;
    bar.innerHTML = "";
    if (!tiers || !tiers.length) return;
    tiers.forEach(function (t) {
      var btn = document.createElement("button");
      btn.type = "button";
      var active = t.tier_id === selectedTier;
      btn.className = "tier-card-opt" + (active ? " tier-card-opt--active" : "");
      var price = "";
      if (monetization.show_prices && t.price_byn != null) {
        price = '<span class="tier-card-opt__price">' + escapeHtml(String(t.price_byn)) + " BYN</span>";
      }
      var hint = t.hint_ru
        ? '<span class="tier-card-opt__hint">' + escapeHtml(t.hint_ru) + "</span>"
        : "";
      btn.innerHTML =
        '<span class="tier-card-opt__head">' +
        '<span class="tier-card-opt__label">' + escapeHtml(t.label_ru || t.tier_id) + "</span>" +
        price +
        "</span>" + hint;
      btn.addEventListener("click", function () {
        selectedTier = t.tier_id;
        renderTierBar(tiers);
        updateBtn();
        track("tier_pick", { tier: t.tier_id });
      });
      bar.appendChild(btn);
    });
  }

  function syncPatientMonetizationFromApi() {
    fetch(window.location.origin + "/api/patient/status")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data) return;
        var mon = data.monetization || {};
        if (!mon.tiers && data.tiers) mon.tiers = data.tiers;
        mon.payment_required = !!data.payment_required;
        applyMonetizationState(mon);
      })
      .catch(function () {});
  }

  function needsPaymentBeforeReview() {
    return !!(monetization.payment_required && !paidToken);
  }

  function startPaymentSession(thenReview) {
    statusEl.textContent = "Создаём сессию оплаты…";
    fetch(window.location.origin + "/api/patient/payment/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tier_id: selectedTier || "basic", clinic_id: clinicId || null }),
    })
      .then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j.detail || "Ошибка оплаты"); return j; }); })
      .then(function (sess) {
        if (sess.payment_token) {
          paidToken = sess.payment_token;
          localStorage.setItem("protocol_patient_payment_token", paidToken);
        }
        if (sess.provider === "dev-mock" && paidToken) {
          statusEl.textContent = "Оплата (демо) принята. Запускаем проверку…";
          if (useSse) runReviewSse(); else runReviewFetch();
          return;
        }
        if (thenReview && paidToken) {
          statusEl.textContent = "";
          if (useSse) runReviewSse(); else runReviewFetch();
          return;
        }
        if (sess.payment_url) {
          window.location.href = sess.payment_url;
        } else {
          statusEl.textContent = "Оплата недоступна. Обратитесь в клинику.";
          updateBtn();
        }
      })
      .catch(function (err) {
        statusEl.textContent = err.message || "Не удалось начать оплату.";
        updateBtn();
      });
  }

  function ensureGuestSession() {
    if (localStorage.getItem(SESSION_KEY)) return;
    fetch(window.location.origin + "/api/patient/account/session", { method: "POST" })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.session_token) localStorage.setItem(SESSION_KEY, data.session_token);
      }).catch(function () {});
  }

  if (paidToken) localStorage.setItem("protocol_patient_payment_token", paidToken);
  if (paidToken && statusEl) statusEl.textContent = "Оплата подтверждена. Загрузите КЗ и нажмите «Проверить».";
  loadClinic();
  syncPatientMonetizationFromApi();
  ensureGuestSession();
  loadQuestionTone();
  renderTonePicker();
  syncUploadFormatsFromApi();
  syncQuestionTonesFromApi();
  var btnJumpQ = document.getElementById("btn-jump-questions");
  if (btnJumpQ) {
    btnJumpQ.addEventListener("click", function () {
      var t = document.getElementById("questions-section");
      if (t) t.scrollIntoView({ behavior: "smooth", block: "start" });
      track("result_jump", { target: "questions" });
    });
  }
  var btnJumpV = document.getElementById("btn-jump-visit");
  if (btnJumpV) {
    btnJumpV.addEventListener("click", function () {
      var t = document.getElementById("visit-sheet-wrap");
      if (t) t.scrollIntoView({ behavior: "smooth", block: "start" });
      track("result_jump", { target: "visit_sheet" });
    });
  }
  if (!restoreReport()) updateBtn();

  function refreshPatientShell() {
    if (!("serviceWorker" in navigator)) return;
    navigator.serviceWorker.getRegistrations().then(function (regs) {
      regs.forEach(function (r) { r.update(); });
    });
    var meta = document.querySelector('meta[name="protocol-patient-build"]');
    var build = meta && meta.getAttribute("content");
    if (build && build.indexOf("__BUILD") === 0) return;
    try {
      var prev = localStorage.getItem("protocol_patient_build_seen");
      if (prev && prev !== build) {
        sessionStorage.removeItem(REPORT_KEY);
        sessionStorage.removeItem("protocol_patient_last_report_v2");
      }
      if (build) localStorage.setItem("protocol_patient_build_seen", build);
    } catch (e) {}
  }
  refreshPatientShell();
  if ("serviceWorker" in navigator) {
    var buildMeta = document.querySelector('meta[name="protocol-patient-build"]');
    var buildTag = buildMeta && buildMeta.getAttribute("content");
    var swUrl = "/patient-sw.js";
    if (buildTag && buildTag.indexOf("__BUILD") !== 0) swUrl += "?v=" + encodeURIComponent(buildTag);
    navigator.serviceWorker.register(swUrl).catch(function () {});
  }
})();
