# Handoff: MO drawer zoom 200%

Дата: 2026-09-07

- Branch: `cursor/mo-drawer-zoom-overflow-agent1-pc1`
- Base: `acc0d2cc9727f0e857cb880af4526d8a49eff3d8`
- BUILD_VERSION: `2026-09-07-081946Z-drawer-zoom`
- PR: [#241](https://github.com/akuazuk/protocol/pull/241)

## Реализация

Для эффективной ширины 180-240 CSS px drawer header, navigation buttons,
tabs и body используют minmax columns, reduced padding и wrapping. Это
соответствует экрану 320-480 px при browser zoom 200%.

## Проверки

- E22 viewport 320/360/768/1024/1440: успешно.
- Effective 180 px, keyboard-only, long synthetic text, tab switch и отсутствие
  horizontal overflow внутри dialog: успешно.
- UI dash normalization и diff-check: успешно.

## Production baseline

- SHA: `acc0d2cc9727f0e857cb880af4526d8a49eff3d8`
- Version: `2026-09-07-073641Z-acceptance-guards`

Следующий шаг: required CI → merge → exact-main GCE deploy → Wave E test-only PR.
