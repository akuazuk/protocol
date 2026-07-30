(function (MO) {
  "use strict";

  var registry = new Map();
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  function token(name, fallback) {
    var value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  }

  function themedOption(option, config) {
    option = Object.assign({}, option || {});
    option.aria = Object.assign({ enabled: true, decal: { show: true } }, option.aria || {});
    option.animation = !reduceMotion.matches;
    option.animationDuration = reduceMotion.matches ? 0 : 180;
    option.textStyle = Object.assign({
      color: token("--ink", "#18312c"),
      fontFamily: token("--font-ui", "system-ui")
    }, option.textStyle || {});
    option.backgroundColor = "transparent";
    if (option.xAxis) {
      option.xAxis = Object.assign({
        axisLabel: { color: token("--muted", "#5f716d") },
        axisLine: { lineStyle: { color: token("--line", "#dbe5e1") } }
      }, option.xAxis);
    }
    if (option.yAxis) {
      option.yAxis = Object.assign({
        axisLabel: { color: token("--muted", "#5f716d") },
        splitLine: { lineStyle: { color: token("--line", "#dbe5e1") } }
      }, option.yAxis);
    }
    option.color = option.color || [
      token("--accent", "#11715e"),
      token("--warn", "#a15c00"),
      token("--bad", "#b4233c"),
      token("--good", "#147a57")
    ];
    if (config && config.description) option.aria.description = config.description;
    return option;
  }

  function dispose(element) {
    var entry = registry.get(element);
    if (!entry) return;
    if (entry.observer) entry.observer.disconnect();
    if (entry.resize) window.removeEventListener("resize", entry.resize);
    entry.chart.dispose();
    registry.delete(element);
  }

  function moChart(element, option, config) {
    config = config || {};
    if (!element) return null;
    dispose(element);
    if (!window.echarts || typeof window.echarts.init !== "function") {
      element.classList.add("chart-fallback");
      if (typeof config.fallback === "function") config.fallback(element);
      return null;
    }
    element.classList.remove("chart-fallback");
    element.classList.add("mo-chart");
    element.setAttribute("role", "img");
    if (config.label) element.setAttribute("aria-label", config.label);
    var chart = window.echarts.init(element, null, { renderer: config.renderer || "svg" });
    chart.setOption(themedOption(option, config), true);
    var resize = function () { if (!chart.isDisposed()) chart.resize(); };
    var observer = null;
    if ("ResizeObserver" in window) {
      observer = new ResizeObserver(resize);
      observer.observe(element);
    } else {
      window.addEventListener("resize", resize, { passive: true });
    }
    registry.set(element, { chart: chart, observer: observer, resize: observer ? null : resize });
    return chart;
  }

  function exportChartPng(element, filename) {
    var entry = registry.get(element);
    if (!entry) return false;
    var link = document.createElement("a");
    link.href = entry.chart.getDataURL({
      type: "png",
      pixelRatio: 2,
      backgroundColor: token("--surface", "#ffffff")
    });
    link.download = filename || "mo-chart.png";
    document.body.appendChild(link);
    link.click();
    link.remove();
    return true;
  }

  MO.moChart = moChart;
  MO.exportChartPng = exportChartPng;
  MO.disposeChart = dispose;
})(window.MO = window.MO || {});
