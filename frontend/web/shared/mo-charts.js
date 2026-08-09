(function (MO) {
  "use strict";

  var registry = new Map();
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  function token(name, fallback) {
    var value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  }

  function mergeDeep(base, override) {
    var result = Object.assign({}, base || {});
    Object.keys(override || {}).forEach(function (key) {
      var left = result[key];
      var right = override[key];
      if (
        left && right &&
        typeof left === "object" && typeof right === "object" &&
        !Array.isArray(left) && !Array.isArray(right)
      ) {
        result[key] = mergeDeep(left, right);
      } else if (right !== undefined) {
        result[key] = right;
      }
    });
    return result;
  }

  function themeAxis(axis, defaults) {
    if (!axis) return defaults;
    if (Array.isArray(axis)) {
      return axis.map(function (item) { return themeAxis(item, defaults); });
    }
    return mergeDeep(defaults, axis);
  }

  function themedOption(option, config) {
    option = Object.assign({}, option || {});
    option.aria = Object.assign({ enabled: true, decal: { show: true } }, option.aria || {});
    option.animation = !reduceMotion.matches;
    option.animationDuration = reduceMotion.matches ? 0 : 420;
    option.animationEasing = "cubicOut";
    option.textStyle = Object.assign({
      color: token("--ink", "#14241f"),
      fontFamily: token("--font-ui", "system-ui"),
      fontSize: 12
    }, option.textStyle || {});
    option.backgroundColor = "transparent";
    option.tooltip = mergeDeep({
      backgroundColor: token("--surface-raised", "#ffffff"),
      borderColor: token("--line", "#d7e3de"),
      borderWidth: 1,
      axisPointer: { type: "cross" },
      textStyle: { color: token("--ink", "#14241f"), fontSize: 12 },
      extraCssText: "border-radius:12px;box-shadow:0 10px 24px rgba(22,40,36,.08);padding:10px 12px;"
    }, option.tooltip || {});
    option.legend = mergeDeep({
      textStyle: { color: token("--muted", "#5b6f6a") },
      icon: "roundRect",
      itemWidth: 12,
      itemHeight: 8
    }, option.legend || {});
    option.grid = mergeDeep({
      containLabel: true,
      left: 18,
      right: 18,
      top: 48,
      bottom: 28
    }, option.grid || {});
    option.toolbox = mergeDeep({
      right: 8,
      top: 6,
      itemSize: 14,
      feature: {
        restore: { title: "Сброс" },
        saveAsImage: { title: "Скачать PNG", pixelRatio: 2 }
      }
    }, option.toolbox || {});

    var axisDefaults = {
      axisLabel: { color: token("--muted", "#5b6f6a"), hideOverlap: true },
      axisLine: { lineStyle: { color: token("--line", "#d7e3de") } },
      axisTick: { show: false },
      splitLine: { lineStyle: { color: token("--line-soft", "#ebf1ee"), type: "dashed" } }
    };
    if (option.xAxis) option.xAxis = themeAxis(option.xAxis, axisDefaults);
    if (option.yAxis) option.yAxis = themeAxis(option.yAxis, axisDefaults);

    option.color = option.color || [
      token("--chart-1", token("--accent", "#2f6f63")),
      token("--chart-2", "#4a6fa5"),
      token("--chart-3", "#a67c52"),
      token("--chart-4", "#8a6b7a"),
      token("--chart-5", "#6b7a8f")
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
    var chart = window.echarts.init(element, null, { renderer: config.renderer || "canvas" });
    try {
      chart.setOption(themedOption(option, config), true);
    } catch (error) {
      element.classList.add("chart-fallback");
      if (typeof config.fallback === "function") config.fallback(element);
      else element.innerHTML = '<div class="empty">Не удалось построить график</div>';
      try { chart.dispose(); } catch (disposeError) {}
      return null;
    }
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
    if (!entry || !entry.chart || typeof entry.chart.getDataURL !== "function") return false;
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
  MO.themeAxis = themeAxis;
})(window.MO = window.MO || {});
