(function (MO) {
  "use strict";

  var TOKEN_KEY = "protocol_methodist_token";
  var ROLE_KEY = "protocol_methodist_role";
  var ADMIN_TOKEN_KEY = "protocol_methodist_admin_token";
  var API_ROOT = "/api/methodist/mo";
  var LEGACY_ROOT = "/api/methodist/mis-kz-quality";

  function token() {
    try {
      return sessionStorage.getItem(TOKEN_KEY) || localStorage.getItem(TOKEN_KEY) || "";
    } catch (error) {
      return "";
    }
  }

  function headers() {
    var result = { Accept: "application/json" };
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
    headers: headers,
    request: request,
    token: token
  });
})(window.MO = window.MO || {});
