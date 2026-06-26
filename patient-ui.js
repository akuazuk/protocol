(function () {
  "use strict";

  var HISTORY_KEY = "protocol_patient_history_v2";
  var CHECKLIST_KEY = "protocol_patient_checklist_v1";
  var ONBOARD_KEY = "protocol_patient_onboard_done";
  var REPORT_KEY = "protocol_patient_last_report_v2";
  var REMINDER_KEY = "protocol_patient_reminder_v1";
  var SESSION_KEY = "protocol_patient_session_token";

  var params = new URLSearchParams(window.location.search);
  var clinicId = params.get("clinic") || "";
  var tierId = params.get("tier") || "";
  var paidToken = params.get("paid") || localStorage.getItem("protocol_patient_payment_token") || "";

  var lastReport = null;
  var clinicConfig = null;
  var selectedTier = tierId || "basic";
  var useSse = true;

  var kzInput = document.getElementById("kz-files");
  var labInput = document.getElementById("lab-files");
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

  function renderProtocolLink(link, fallbackTitle) {
    if (!link || !link.pdf_url) {
      return fallbackTitle ? escapeHtml(fallbackTitle) : "";
    }
    var title = link.title || fallbackTitle || "Клинический протокол Минздрава";
    return '<a class="proto-link" href="' + escapeHtml(link.pdf_url) + '" target="_blank" rel="noopener noreferrer" title="Открыть PDF протокола">' + escapeHtml(title) + "</a>";
  }

  function renderProtocolLinksList(links, limit) {
    if (!links || !links.length) return "";
    var n = limit || links.length;
    var parts = [];
    for (var i = 0; i < links.length && i < n; i++) {
      parts.push(renderProtocolLink(links[i]));
    }
    return parts.join(", ");
  }

  function track(event, meta) {
    try {
      fetch(window.location.origin + "/api/patient/analytics", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ event: event, clinic_id: clinicId || null, tier_id: selectedTier, meta: meta || {} }),
      }).catch(function () {});
    } catch (e) {}
  }

  function renderChips(input, containerId) {
    var el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML = "";
    if (!input.files || !input.files.length) return;
    for (var i = 0; i < input.files.length; i++) {
      var span = document.createElement("span");
      span.className = "file-chip";
      span.textContent = input.files[i].name;
      el.appendChild(span);
    }
  }

  function updateBtn() {
    btn.disabled = !(consentEl.checked && kzInput.files && kzInput.files.length);
  }

  if (kzInput) kzInput.addEventListener("change", function () { renderChips(kzInput, "kz-chips"); updateBtn(); });
  if (labInput) labInput.addEventListener("change", function () { renderChips(labInput, "lab-chips"); });
  if (consentEl) consentEl.addEventListener("change", updateBtn);

  if (localStorage.getItem(ONBOARD_KEY) === "1" && onboard) onboard.classList.add("hidden");
  var btnOnboard = document.getElementById("btn-onboard-ok");
  if (btnOnboard) btnOnboard.addEventListener("click", function () {
    localStorage.setItem(ONBOARD_KEY, "1");
    if (onboard) onboard.classList.add("hidden");
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
    try { sessionStorage.setItem(REPORT_KEY, JSON.stringify(pr)); } catch (e) {}
  }
  function restoreReport() {
    try {
      var raw = sessionStorage.getItem(REPORT_KEY);
      if (!raw) return false;
      var pr = JSON.parse(raw);
      if (!pr || !pr.plain_summary_ru) return false;
      formCard.classList.add("hidden");
      resultCard.classList.remove("hidden");
      renderReport(pr);
      track("restore_report", { light: pr.traffic_light });
      return true;
    } catch (e) { return false; }
  }

  function renderScoreRing(pct, light, label, compact) {
    var svg = document.getElementById("score-svg");
    var card = document.getElementById("score-card-wrap");
    if (!svg) return;
    if (card) card.classList.toggle("score-card--secondary", !!compact);
    var r = compact ? 40 : 52;
    var c = 2 * Math.PI * r;
    var off = c * (1 - (pct != null ? pct / 100 : 0));
    var color = light === "green" ? "#1a8a72" : light === "red" ? "#dc2626" : "#d97706";
    var cx = compact ? 44 : 60;
    svg.setAttribute("viewBox", compact ? "0 0 88 88" : "0 0 120 120");
    svg.innerHTML =
      '<circle cx="' + cx + '" cy="' + cx + '" r="' + r + '" fill="none" stroke="#e8f5f1" stroke-width="' + (compact ? 8 : 10) + '"/>' +
      '<circle cx="' + cx + '" cy="' + cx + '" r="' + r + '" fill="none" stroke="' + color + '" stroke-width="' + (compact ? 8 : 10) + '" ' +
      'stroke-dasharray="' + c + '" stroke-dashoffset="' + off + '" stroke-linecap="round" transform="rotate(-90 ' + cx + " " + cx + ')"/>' +
      '<text x="' + cx + '" y="' + (cx - 2) + '" text-anchor="middle" font-size="' + (compact ? "16" : "22") + '" font-weight="800" fill="#063d35">' +
      (pct != null ? pct + "%" : "—") + "</text>";
    var cap = document.getElementById("score-caption");
    if (cap) cap.textContent = label || "";
  }

  function renderTrafficPill(light, label) {
    var el = document.getElementById("traffic-pill");
    if (!el) return;
    el.className = "traffic-pill traffic-pill--" + (light === "green" ? "green" : light === "red" ? "red" : "yellow");
    el.innerHTML = '<span aria-hidden="true">' + trafficIcon(light) + "</span> " + escapeHtml(label || "");
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
        var val = r.value != null && r.value !== "" ? String(r.value) : "—";
        if (r.unit) val += " " + r.unit;
        if (r.flag === "high") val += " ↑";
        html += "<tr><td>" + escapeHtml(r.marker || "—") + "</td><td>" + escapeHtml(val) + "</td><td>" + (r.in_kz ? "да" : "нет") + "</td></tr>";
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
    var html = '<details class="cites-fold"><summary>Требования протокола Минздрава</summary><div style="margin-top:0.45rem">';
    if (pc.protocol_title || pc.protocol_link) {
      html += '<p style="font-size:0.78rem;color:var(--muted);margin:0 0 0.5rem">';
      if (pc.protocol_link) html += "Протокол: " + renderProtocolLink(pc.protocol_link, pc.protocol_title);
      else html += escapeHtml(pc.protocol_title || "");
      html += "</p>";
    }
    pc.missing_recommended_exams.forEach(function (m) {
      html += '<div class="block-card block-card--concern" style="margin-bottom:0.4rem"><strong>' + escapeHtml(m.exam_name || "Обследование") + "</strong>";
      html += "<p style=\"margin:0.25rem 0 0\">" + escapeHtml(m.patient_note_ru || "") + "</p></div>";
    });
    html += "</div></details>";
    box.innerHTML = html;
  }

  function saveHistory(pr) {
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
        return '<div class="history-item">' + d.toLocaleString("ru-RU") + " · " + (it.pct != null ? it.pct + "%" : "—") + " · " + escapeHtml(it.label || "") + "</div>";
      }).join("");
    } catch (e) { wrap.classList.add("hidden"); }
  }
  renderHistory();

  function buildShareText(pr) {
    var lines = ["Проверь КЗ — лист на приём", ""];
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

  function renderReport(pr) {
    lastReport = pr;
    saveReport(pr);

    var hl = document.getElementById("headline-ru");
    if (hl) hl.textContent = pr.headline_ru || pr.overall_label_ru || "";

    renderTrafficPill(pr.traffic_light, pr.overall_label_ru);
    renderQualityBanner(pr.document_quality, pr.traffic_light);
    renderScoreRing(pr.overall_pct, pr.traffic_light, "Детальная оценка по блокам", true);

    var mb = document.getElementById("matched-badge");
    var cnt = pr.matched_protocols_count;
    if (mb) {
      if (cnt > 0 || (pr.protocol_links && pr.protocol_links.length)) {
        var links = pr.protocol_links || [];
        if (links.length) {
          mb.innerHTML = "Сверка с " + renderProtocolLinksList(links, 3);
          if (links.length > 3) mb.innerHTML += " и ещё " + (links.length - 3);
        } else {
          mb.textContent = "Сверка с " + cnt + " протоколами Минздрава";
        }
        mb.classList.remove("hidden");
      } else mb.classList.add("hidden");
    }

    var ps = document.getElementById("plain-summary");
    if (ps) ps.textContent = pr.plain_summary_ru || "";

    var ns = document.getElementById("next-steps");
    if (ns) {
      ns.innerHTML = "";
      (pr.next_steps_ru || []).forEach(function (s) {
        var li = document.createElement("li");
        li.textContent = s;
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

    var cl = document.getElementById("action-checklist");
    if (cl) {
      cl.innerHTML = "";
      var checklistState = loadChecklistState();
      var items = pr.action_checklist || [];
      items.forEach(function (item) {
        var li = document.createElement("li");
        var checked = !!checklistState[item.id];
        if (checked) li.className = "checked";
        li.innerHTML = '<input type="checkbox" id="ck-' + escapeHtml(item.id) + '" ' + (checked ? "checked" : "") + " />" +
          '<label for="ck-' + escapeHtml(item.id) + '"><span class="checklist__title">' + escapeHtml(item.title || "Вопрос") + "</span>" + escapeHtml(item.text || "") + "</label>";
        var cb = li.querySelector("input");
        cb.addEventListener("change", function () {
          saveChecklistItem(item.id, cb.checked);
          li.classList.toggle("checked", cb.checked);
        });
        cl.appendChild(li);
      });
    }

    var cards = document.getElementById("block-cards");
    if (cards) {
      cards.innerHTML = "";
      (pr.blocks || []).forEach(function (b) {
        var div = document.createElement("div");
        div.className = "block-card block-card--" + (b.status || "attention");
        var gaps = b.gaps && b.gaps.length ? "<ul class=\"gap-list\">" + b.gaps.map(function (g) { return "<li>" + escapeHtml(g) + "</li>"; }).join("") + "</ul>" : "";
        var why = b.why_ru ? '<p class="block-card__why"><span class="block-card__why-label">Почему так: </span>' + escapeHtml(b.why_ru) + "</p>" : "";
        var protoLine = "";
        if (b.protocol_excerpt || b.protocol_link) {
          protoLine = '<p class="block-card__proto"><span class="block-card__proto-label">По </span>';
          if (b.protocol_link) protoLine += renderProtocolLink(b.protocol_link);
          else protoLine += "протоколу";
          if (b.protocol_excerpt) protoLine += ": " + escapeHtml(b.protocol_excerpt);
          protoLine += "</p>";
        }
        div.innerHTML =
          '<div class="block-card__head"><span class="block-card__title">' + escapeHtml(b.title) + "</span>" +
          '<span class="' + pillClass(b.status) + ' block-card__score">' +
          (b.score_pct != null ? b.score_pct + "% · " : "") + pillLabel(b.status) + "</span></div>" +
          '<div class="block-card__comment"><span class="block-card__comment-label">Комментарий</span>' +
          escapeHtml(b.summary_ru || "—") + why + protoLine + gaps + "</div>";
        cards.appendChild(div);
      });
    }

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
          var head = renderProtocolLink(c.protocol_link, c.protocol_title || "Протокол");
          if (c.section) head += " · " + escapeHtml(c.section);
          div.innerHTML = "<strong>" + head + "</strong>" + escapeHtml(c.excerpt || "");
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
    track("report_view", { light: pr.traffic_light, pct: pr.overall_pct, block_count: (pr.blocks || []).length });
  }

  function showLoader(stage) {
    loader.classList.remove("hidden");
    if (loaderText) loaderText.textContent = stage || "Анализируем документ";
  }
  function hideLoader() { loader.classList.add("hidden"); }

  function buildFormData() {
    var fd = new FormData();
    var i;
    for (i = 0; i < kzInput.files.length; i++) fd.append("files", kzInput.files[i]);
    if (labInput.files) for (i = 0; i < labInput.files.length; i++) fd.append("lab_files", labInput.files[i]);
    fd.append("consent", "1");
    var age = document.getElementById("age-years");
    var sex = document.getElementById("sex");
    if (age && age.value) fd.append("age_years", age.value);
    if (sex && sex.value) fd.append("sex", sex.value);
    if (clinicId) fd.append("clinic_id", clinicId);
    if (selectedTier) fd.append("tier_id", selectedTier);
    if (paidToken) fd.append("payment_token", paidToken);
    return fd;
  }

  function handleReviewResult(data) {
    hideLoader();
    if (btn) btn.disabled = false;
    formCard.classList.add("hidden");
    resultCard.classList.remove("hidden");
    renderReport((data && data.patient_report) || {});
    window.scrollTo({ top: 0, behavior: "smooth" });
    track("upload_done", { light: (data.patient_report || {}).traffic_light, latency_ms: data.latency_ms });
  }

  function runReviewFetch() {
    track("upload_start", { tier: selectedTier, lab_count: labInput.files ? labInput.files.length : 0 });
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
    if (!kzInput.files || !kzInput.files.length) return;
    if (useSse) runReviewSse(); else runReviewFetch();
  });

  var btnPrint = document.getElementById("btn-print");
  if (btnPrint) btnPrint.addEventListener("click", function () { track("print_tap"); window.print(); });

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
    resultCard.classList.add("hidden");
    formCard.classList.remove("hidden");
    kzInput.value = "";
    labInput.value = "";
    renderChips(kzInput, "kz-chips");
    renderChips(labInput, "lab-chips");
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
      var brand = document.querySelector(".brand");
      if (brand && clinicId) brand.textContent = clinicConfig.name_ru;
      if (!tierId && clinicConfig.default_tier) selectedTier = clinicConfig.default_tier;
      loadTiers();
    }).catch(function () {});
  }

  function loadTiers() {
    fetch(window.location.origin + "/api/patient/tiers").then(function (r) { return r.json(); }).then(function (data) {
      var bar = document.getElementById("tier-bar");
      if (!bar || !data.tiers) return;
      bar.innerHTML = "";
      data.tiers.forEach(function (t) {
        var chip = document.createElement("button");
        chip.type = "button";
        chip.className = "tier-chip" + (t.tier_id === selectedTier ? " tier-chip--active" : "");
        chip.textContent = t.label_ru + " · " + t.price_byn + " BYN";
        chip.addEventListener("click", function () {
          selectedTier = t.tier_id;
          loadTiers();
        });
        bar.appendChild(chip);
      });
    }).catch(function () {});
  }

  function ensureGuestSession() {
    if (localStorage.getItem(SESSION_KEY)) return;
    fetch(window.location.origin + "/api/patient/account/session", { method: "POST" })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (data.session_token) localStorage.setItem(SESSION_KEY, data.session_token);
      }).catch(function () {});
  }

  function setupInstallHint() {
    var hint = document.getElementById("install-hint");
    if (!hint || window.matchMedia("(display-mode: standalone)").matches) return;
    hint.classList.remove("hidden");
    var btnInstall = document.getElementById("btn-install-dismiss");
    if (btnInstall) btnInstall.addEventListener("click", function () { hint.classList.add("hidden"); });
  }

  if (paidToken) localStorage.setItem("protocol_patient_payment_token", paidToken);
  loadClinic();
  loadTiers();
  ensureGuestSession();
  setupInstallHint();
  if (!restoreReport()) updateBtn();

  if ("serviceWorker" in navigator) navigator.serviceWorker.register("/patient-sw.js").catch(function () {});

  var deferredPrompt;
  window.addEventListener("beforeinstallprompt", function (e) {
    e.preventDefault();
    deferredPrompt = e;
    var ib = document.getElementById("btn-install-pwa");
    if (ib) {
      ib.classList.remove("hidden");
      ib.addEventListener("click", function () {
        if (deferredPrompt) deferredPrompt.prompt();
        track("install_pwa_prompt");
      });
    }
  });
})();
