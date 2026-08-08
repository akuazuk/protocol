# Handoff: GCE HTTPS (Caddy) — 2026-08-07

## Repo
- worktree: `/private/tmp/protocol-task-gce-https-caddy-pc1`
- branch: `cursor/gce-https-caddy-pc1`
- base: `origin/main` @ `1520d61e`
- HEAD: see PR
- PR: open from this branch

## Done
- Caddy 2.11.4 on `protocol-app` → `:8000`
- DNS A at hoster.by → `34.118.21.47` (CNAME removed)
- LE cert issued (CN=`protocol.kravira.by`, YE2, ~90 days)
- Smoke via GCE IP / 8.8.8.8: `/health/live` + `/api/version` ok
- PR #46

## Not done
- Some resolvers still cache old CNAME→Render (TTL up to ~1h); flush or wait
- Secret Manager still open
- Render remains at `https://protocol-bimy.onrender.com`

## Verify
```bash
dig @u1.hoster.by +short protocol.kravira.by A   # 34.118.21.47
dig @8.8.8.8 +short protocol.kravira.by A
curl -fsS https://protocol.kravira.by/health/live
curl -fsS https://protocol.kravira.by/api/version
```

## Do not touch in parallel
- `deploy/gcp-app/*` HTTPS/DNS, GCE firewall, Render custom domain for `protocol.kravira.by`
