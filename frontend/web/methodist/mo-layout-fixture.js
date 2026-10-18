/* Учебные визиты для сверки вёрстки. Сеть не вызывается. */
(function () {
  "use strict";
  var EXAM = "Текст осмотра фикстуры";
  var visits = [
    {
      key: "fx-major",
      grade: "important",
      gradeLabel: "Важно",
      reason: "Major: Препарат А + Препарат Б",
      badge: "Major"
    },
    {
      key: "fx-nsaid",
      grade: "important",
      gradeLabel: "Важно",
      reason: "Одновременно два НПВП"
    },
    {
      key: "fx-empty-dx",
      grade: "important",
      gradeLabel: "Важно",
      reason: "Диагноз пустой, код МКБ не указан"
    },
    {
      key: "fx-stale",
      grade: "important",
      gradeLabel: "Важно",
      reason: "Оценка старше замечания: живого сигнала P0/P1 нет",
      stale: true
    },
    {
      key: "fx-critical",
      grade: "critical",
      gradeLabel: "Критично",
      reason: "Красный флаг: учебный сигнал Альфа"
    }
  ];
  window.MO_LAYOUT_FIXTURE = { exam: EXAM, visits: visits };
  document.documentElement.setAttribute("data-fixture-ready", "1");
})();
