# Единый вход и учётки: методист + МО Аналитика (v1)

Дата: 2026-08-09  
Статус: active  
Связанные: `2026-08-05-mo-expert-reviewer-portal-v1.md` (архив; expert UI выводится),
`2026-07-28-mo-daily-bi-platform-v1.md` (CRM SQLite / roles).

---

## 1. Контекст

Сейчас три параллельных контура входа:

| Контур | Auth | Storage | Scope |
|--|--|--|--|
| Кабинет методиста `?mode=methodist` | общий `METHODIST_TOKEN` | только `sessionStorage` | КЗ / AI-review |
| МО Аналитика `/methodist/mo` | тот же токен | `localStorage` + `sessionStorage` | полный BI |
| Кабинет эксперта `/methodist/expert` | login/password → session 12ч | `protocol_expert_session` | только вчера + отчёты |

Проблемы:

1. SSO кабинет ↔ МО ломается на `target="_blank"` (новый tab без `sessionStorage`).
2. Любой HTTP 403 в МО кидает на экран входа, даже при живой сессии.
3. Роль `expert` дублирует методиста и режет BI; нужна одна роль методиста.
4. Персональных учёток методиста нет: один shared secret на всех.

Цель: стабильный вход методиста в оба раздела; затем login/password + админка уровней.

---

## 2. Целевая модель

Один вход: **логин + пароль → серверная сессия** (SQLite warehouse), общая для
кабинета методиста и МО Аналитики.

Роли (без `expert`):

| Роль | Доступ |
|--|--|
| `methodist` | кабинет + полный МО BI + review pack |
| `lead` | methodist + расширенные действия |
| `admin` | lead + CRUD учёток |
| `viewer` (опц. позже) | только чтение |

`METHODIST_TOKEN` остаётся ops/bootstrap для скриптов и аварийного входа, не основной UX.

Каркас: обобщить `mo_expert_auth` → `crm_app_user` / `crm_app_session` с полем `role`.

---

## 3. Метрики

| Метрика | Было | Цель P0 | Цель P1+ |
|--|--|--|--|
| Вход методист → МО без повторного токена (новая вкладка) | нет | да | да |
| МО → кабинет методиста без повторного токена | нет | да | да |
| Ложный logout из‑за 403 «нет прав» | да | нет | нет |
| Отдельный UI эксперта в проде | да | redirect на `/methodist/mo` | удалён |
| Персональные учётки методиста | нет | нет | да + admin CRUD |

---

## 4. Шаги

### P0 - стабильность shared-token SSO (эта волна)

- [x] План в `docs/plans/` + индекс.
- [x] Кабинет методиста: читать/писать `protocol_methodist_token` (+ reviewer) в
  `localStorage` и `sessionStorage`.
- [x] `mo-api.js`: единые `setToken` / `clearToken`; на странице методиста не
  предпочитать leftover expert-session поверх methodist-token.
- [x] `mo-app.js`: 401/403 → экран входа только при признаках auth-failure;
  иначе toast «недостаточно прав».
- [x] `/methodist/expert*` → 302 на `/methodist/mo`.
- [x] Тесты маршрута + статические маркеры storage/auth.
- [x] BUILD_VERSION, PR.

### P1 - login/password для методиста

- [x] Обобщить expert-auth → app-users с ролями `methodist|lead|admin`.
- [x] UI входа логин/пароль в кабинете и МО (вместо поля токена как основного).
- [x] Одна session на origin; sliding TTL 12–24ч.
- [ ] Миграция: bootstrap admin из env; опционально перенос expert users → methodist.

### P2 - админка учёток

- [x] `/methodist/admin` (роль admin).
- [x] CRUD: создать, роль, deactivate, reset password.
- [ ] Аудит входов и изменений учёток.

### P3 - зачистка

- [ ] Удалить expert UI/API/роли из продукта (после миграции данных).
- [ ] `METHODIST_TOKEN` только для CI/скриптов.
- [ ] Archived expert-portal plan закрыт преемником.

---

## 5. Риски

| Риск | Митигация |
|--|--|
| Bookmark `/methodist/expert` | P0 redirect на МО |
| Leftover expert session ломает BI | на non-expert page приоритет methodist-token |
| Shared token всё ещё один на всех | P1 персональные учётки |
| SQLite sessions на Render disk | уже паттерн expert; не трогать CRM при publish |

---

## 6. Решения владельца

1. Эксперта удаляем из UX (P0 redirect); данные/API вычищаем в P3.
2. Стартуем с P0 (storage + 403), затем P1 login/password.
3. Первый admin: bootstrap из env (`MO_ADMIN_BOOTSTRAP_*`) в P1.
