# Handoff: MO settings → Справка

- repo: `akuazuk/protocol`
- branch: `cursor/mo-settings-refresh-pc1`
- worktree: `/private/tmp/protocol-task-mo-settings-refresh-pc1`
- base: `origin/main` @ `c99a607c`
- HEAD: `5bb4e9fb`
- PR: https://github.com/akuazuk/protocol/pull/93
- BUILD_VERSION: `2026-08-09-075827Z-mo-settings-refresh`

## Done

- Sidebar foot **Справка** → settings page
- Removed v3/v4 switcher and AI expenses UI
- Zones legend + session + about; density/views kept
- Plan + UI static tests

## Not done

- Merge/deploy (await review; sync with #92)
- Production smoke after Action deploy

## Next

```bash
gh pr view 93 --repo akuazuk/protocol
# after merge of #92 or #93: rebase the other onto origin/main
```

## Parallel touch

Do not edit in parallel with #92: `frontend/web/methodist/mis-kz-quality.html`, `frontend/web/shared/mo-app.js`.
