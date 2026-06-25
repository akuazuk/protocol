# B2C: мобильное приложение «Проверь своё заключение»

**Статус:** MVP backend + PWA-витрина · июнь 2026  
**Tier API:** `P1` (L1 structured + alignment, без ЦИСЗ и send_gate)

## Реализовано

| Компонент | Путь |
|-----------|------|
| Сборка отчёта для пациента | `clinical_knowledge/patient_report.py` |
| Оркестрация P1 | `clinical_knowledge/patient_review.py` |
| Сверка анализов (фаза 2a) | `lab_result_parser.py`, `patient_lab_crosscheck.py` |
| API | `POST /api/patient/review` (+ `lab_files`), `/api/patient/review/json`, `/api/patient/status` |
| PWA-витрина | `patient.html`, `patient-manifest.webmanifest`, `patient-sw.js` |
| Ссылка из основного UI | `index.html` → блок «Материалы проекта» и футер |
| Тесты | `tests/test_patient_report.py`, `tests/test_lab_result_parser.py` |

## Переменные окружения

| Переменная | По умолчанию |
|------------|--------------|
| `PATIENT_REVIEW_ENABLED` | `1` |
| `PATIENT_REVIEW_MAX_FILES` | `5` |
| `PATIENT_LAB_MAX_FILES` | `3` |
| `RATE_LIMIT_PATIENT_PER_MIN` | `5` |

## Следующие этапы

1. **React Native / Flutter** - нативная камера, IAP, push.
2. **Аккаунты** - Apple/Google Sign-In, абонемент.
3. **Публикация** - App Store / Google Play.

## Дисклеймер

Ориентировочная сверка с клиническими протоколами Минздрава РБ. Не диагноз, не МЭЭ, не замена очного приёма.
