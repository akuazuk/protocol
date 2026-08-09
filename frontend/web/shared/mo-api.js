(function (MO) {
  "use strict";

  var TOKEN_KEY = "protocol_methodist_token";
  var ROLE_KEY = "protocol_methodist_role";
  var ADMIN_TOKEN_KEY = "protocol_methodist_admin_token";
  var EXPERT_SESSION_KEY = "protocol_expert_session";
  var APP_SESSION_KEY = "protocol_methodist_session";
  var API_ROOT = "/api/methodist/mo";
  var LEGACY_ROOT = "/api/methodist/mis-kz-quality";

  function isExpertAudience() {
    try {
      var meta = document.querySelector('meta[name="mo-audience"]');
      return !!(meta && meta.getAttribute("content") === "expert");
    } catch (error) {
      return false;
    }
  }

  function readStorage(key) {
    try {
      return sessionStorage.getItem(key) || localStorage.getItem(key) || "";
    } catch (error) {
      return "";
    }
  }

  function writeStorage(key, value) {
    var tokenValue = String(value || "").trim();
    try {
      if (tokenValue) {
        sessionStorage.setItem(key, tokenValue);
        localStorage.setItem(key, tokenValue);
      } else {
        sessionStorage.removeItem(key);
        localStorage.removeItem(key);
      }
    } catch (error) {}
  }

  function token() {
    var value = readStorage(TOKEN_KEY);
    // Sync across storages so a new tab (empty sessionStorage) still sees localStorage.
    if (value) writeStorage(TOKEN_KEY, value);
    return value;
  }

  function setToken(value) {
    writeStorage(TOKEN_KEY, value);
  }

  function clearToken() {
    writeStorage(TOKEN_KEY, "");
  }

  function expertToken() {
    return readStorage(EXPERT_SESSION_KEY);
  }

  function setExpertToken(value) {
    writeStorage(EXPERT_SESSION_KEY, value);
  }

  function clearExpertToken() {
    setExpertToken("");
  }

  function appSessionToken() {
    return readStorage(APP_SESSION_KEY);
  }

  function setAppSessionToken(value) {
    writeStorage(APP_SESSION_KEY, value);
  }

  function clearAppSessionToken() {
    setAppSessionToken("");
  }

  function headers() {
    var result = { Accept: "application/json" };
    // On methodist MO pages prefer methodist token so leftover expert session
    // does not silently downgrade / kick BI access.
    if (isExpertAudience()) {
      var expert = expertToken();
      if (expert) {
        result["X-Expert-Session"] = expert;
        return result;
      }
    }
    var appSession = appSessionToken();
    if (appSession) {
      result["X-Methodist-Session"] = appSession;
      return result;
    }
    if (token()) result["X-Methodist-Token"] = token();
    try {
      var role = sessionStorage.getItem(ROLE_KEY) || "";
      var adminToken = sessionStorage.getItem(ADMIN_TOKEN_KEY) || "";
      if (role) result["X-Methodist-Role"] = role;
      if (role === "admin" && adminToken) result["X-Methodist-Admin-Token"] = adminToken;
    } catch (error) {}
    return result;
  }

  async function request(primary, legacy, options) {
    options = options || {};
    options.headers = Object.assign({}, headers(), options.headers || {});
    var response;
    try {
      response = await fetch(API_ROOT + primary, options);
    } catch (error) {
      response = null;
    }
    if (!response || response.status === 404 || response.status === 405 || response.status === 501) {
      if (legacy === "__root__") {
        return fetch("/api/methodist/mis-kz-quality" +
          (primary.indexOf("?") >= 0 ? primary.slice(primary.indexOf("?")) : ""), options);
      }
      return fetch(LEGACY_ROOT + legacy, options);
    }
    return response;
  }

  MO.api = Object.freeze({
    API_ROOT: API_ROOT,
    LEGACY_ROOT: LEGACY_ROOT,
    TOKEN_KEY: TOKEN_KEY,
    ROLE_KEY: ROLE_KEY,
    ADMIN_TOKEN_KEY: ADMIN_TOKEN_KEY,
    EXPERT_SESSION_KEY: EXPERT_SESSION_KEY,
    APP_SESSION_KEY: APP_SESSION_KEY,
    headers: headers,
    request: request,
    token: token,
    setToken: setToken,
    clearToken: clearToken,
    expertToken: expertToken,
    setExpertToken: setExpertToken,
    clearExpertToken: clearExpertToken,
    appSessionToken: appSessionToken,
    setAppSessionToken: setAppSessionToken,
    clearAppSessionToken: clearAppSessionToken,
    isExpertAudience: isExpertAudience
  });
})(window.MO = window.MO || {});
