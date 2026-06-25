# B2C: мобильное приложение «Проверь своё заключение»

**Статус:** MVP backend + веб-витрина (`patient.html`) · июнь 2026  
**Tier API:** `P1` (L1 structured + alignment, без ЦИСЗ и send_gate)

## Реализовано в репозитории

| Компонент | Путь |
|-----------|------|
| Сборка отчёта для пациента | `clinical_knowledge/patient_report.py` |
| Оркестрация P1 | `clinical_knowledge/patient_review.py` |
| API | `POST /api/patient/review`, `POST /api/patient/review/json`, `GET /api/patient/status` |
| Веб-витрина | `patient.html` → `/patient.html`, редирект `/patient` |
| Тесты | `tests/test_patient_report.py` |

## Переменные окружения

| Переменная | По умолчанию | Смысл |
|------------|--------------|--------|
| `PATIENT_REVIEW_ENABLED` | `1` | Включить B2C API |
| `PATIENT_REVIEW_MAX_FILES` | `5` | Фото/PDF за запрос |
| `RATE_LIMIT_PATIENT_PER_MIN` | `5` | Лимит на IP |

## Следующие этапы

1. **React Native / Flutter** - камера, IAP, история проверок.
2. **Анализы** - загрузка бланков, сверка с блоком «обследование».
3. **Аккаунты** - Apple/Google Sign-In, абонемент.
4. **Публикация** - App Store / Google Play, политика ПДн РБ.

## Дисклеймер (обязателен в UI)

Ориентировочная сверка с клиническими протоколами Минздрава РБ. Не диагноз, не МЭЭ, не замена очного приёма.

Полный продуктовый план - в обсуждении с командой (архитектура B2C vs B2B в `docs/architecture-kravira-fhir-mis-print.html`, раздел монетизации в `docs/konkurs/`).
