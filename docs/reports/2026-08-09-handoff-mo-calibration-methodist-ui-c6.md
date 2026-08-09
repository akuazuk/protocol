# Handoff: MO calibration methodist UI C6

Date: 2026-08-09

## Repository state

- repo: `akuazuk/protocol`
- branch: `cursor/mo-calibration-methodist-ui-c6-pc1`
- worktree: `/private/tmp/protocol-task-mo-calibration-methodist-ui-c6-pc1`
- base: `d65fc5e39464f50d384c43b90936e55d8bfd8b75`
- implementation HEAD: `6b390fdb`
- PR: https://github.com/akuazuk/protocol/pull/111
- merge/deploy: not performed

## Completed

- Added a full-width MO-style labeling page at `/methodist/calibration`.
- Linked it from both the methodist cabinet and MO Analytics sidebar.
- Added protected GET/PUT APIs for the frozen C6 review pack.
- Restricted data and writes to methodist, lead, and admin roles.
- Added server-owned reviewer identity and timestamp.
- Added file locking, atomic `0600` label writes, and per-action audit JSONL.
- Added optimistic conflict detection so a stale browser cannot overwrite a
  newer methodist label.
- Added `Cache-Control: no-store` to the page and clinical APIs.
- Kept engine scores, LLM passes, and LLM adjudication out of every UI response.
- Added the final gate: comparison is created only after 22/22 valid human labels.

## Security self-check

- Page HTML is public static shell, but clinical data requires authenticated API.
- Viewer, doctor, and expert roles cannot read or write the calibration API.
- Clinical values are rendered with `textContent`, not HTML interpolation.
- The calibration client never writes clinical cases to browser storage; only
  the existing cabinet auth-token module accesses local/session storage.
- API accepts only pseudonymous `SNNN` sample IDs and fixed endpoint names.
- Verdict, score, harm, ICD fit, confidence, and rationale are server-validated.
- Concurrent writes are serialized with `flock` and replaced atomically.
- Access audit contains actor, role, action, pseudonymous sample, endpoint, time;
  no clinical text or source IDs.

## Verification

- Focused UI/pack/cabinet tests: 12 passed.
- Final calibration, account, auth, contract, and blind regression selection:
  41 passed.
- JavaScript syntax, Python compile, diff check, and IDE lint passed.
- API tests verify authentication, role denial, `no-store`, save, and unseal gate.
- Filesystem tests verify `0600` labels/audit and absence of comparison before gate.

## Current C6 state

- review cases: 18;
- required endpoint labels: 22;
- completed human labels: 0;
- C6 passed: false.

The UI enables the pending work but does not fabricate human gold. C7 remains
blocked until a methodist fills all labels and the server reports `passed=true`.

## Production state

- production scoring/action queue/warehouse changed: no
- production clinical labels changed: no
- deploy: not run
- GCE primary currently remains on
  `2026-08-09-102358Z-methodist-cabinet-style`; deployment needs a designated
  release coordinator after merge.
- GCE has one active `admin:full` account, so an authorized reviewer can use the
  form after deploy; no credential was read or changed.
- Render backup has newer code but does not hold this secret calibration pack.
- `BUILD_VERSION`: `2026-08-09-143534Z-mo-calibration-methodist-ui`

## Safe next step

After merge and the separately coordinated GCE primary deployment, a methodist
signs in and opens:

```text
https://protocol.kravira.by/methodist/calibration
```

The secret pack remains only on GCE. Do not copy it to the Render backup. Do not
start C7 until the progress reaches 22/22.

## Files not to edit in parallel

- `clinical_knowledge/mo_calibration_methodist_ui.py`
- `rag_server.py`
- `frontend/web/methodist/mo-calibration.html`
- `frontend/web/shared/mo-calibration.css`
- `frontend/web/shared/mo-calibration.js`
- `backend/frontend_paths.py`
